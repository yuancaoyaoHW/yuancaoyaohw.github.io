#!/usr/bin/env python3
"""
Fetch latest papers from arXiv by channel configuration.
Uses RSS feeds (no rate limiting) as primary source, API as fallback.
Usage:
  python fetch_papers.py --channel infra --json
  python fetch_papers.py --channel infra --download --json
"""

import json
import subprocess
import re
import argparse
import sys
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "channels.json"

# arXiv API rate limit: 1 request per 3 seconds (conservative)
API_DELAY = 3.5
# Per-request timeout (seconds)
API_TIMEOUT = 60
# Number of retry attempts on failure
MAX_RETRIES = 3


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_rss_category(category):
    """Fetch papers from arXiv RSS feed for a single category.
    RSS feeds have no rate limiting and return all papers from the latest update.
    """
    url = f'https://rss.arxiv.org/rss/{category}'

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ['curl', '-sL', '--max-time', '30', url],
                capture_output=True, timeout=40
            )
            data = result.stdout.decode('utf-8', errors='replace')

            if not data or 'Rate exceeded' in data or len(data.strip()) < 50:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                continue

            return data
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"  [warn] Error fetching RSS {category}: {e}", file=sys.stderr)

    return ""


def parse_rss_items(data, category):
    """Parse RSS XML and extract paper items."""
    papers = []

    items = re.findall(r'<item>(.*?)</item>', data, re.DOTALL)

    for item in items:
        # Extract paper ID from link or guid
        link_m = re.search(r'<link>(.*?)</link>', item)
        if link_m:
            url = link_m.group(1).strip()
            paper_id = url.split('/abs/')[-1].split('v')[0] if '/abs/' in url else ""
        else:
            guid_m = re.search(r'<guid[^>]*>(.*?)</guid>', item)
            if guid_m:
                guid = guid_m.group(1).strip()
                paper_id = guid.split('arXiv.org:')[-1].split('v')[0] if 'arXiv.org:' in guid else ""
            else:
                continue

        if not paper_id:
            continue

        # Title
        title_m = re.search(r'<title>(.*?)</title>', item, re.DOTALL)
        title = title_m.group(1).replace('\n', ' ').strip() if title_m else "No title"

        # Description (contains abstract)
        desc_m = re.search(r'<description>(.*?)</description>', item, re.DOTALL)
        desc = desc_m.group(1) if desc_m else ""
        # RSS description contains "arXiv:IDv1 Announce Type: new \nAbstract: ..."
        summary = ""
        if 'Abstract:' in desc:
            summary = desc.split('Abstract:')[-1].strip()
        elif desc:
            summary = desc.strip()

        # Clean up HTML entities in description
        summary = summary.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')

        # Authors
        creator_m = re.search(r'<dc:creator>(.*?)</dc:creator>', item, re.DOTALL)
        if creator_m:
            authors_str = creator_m.group(1).strip()
            authors = [a.strip() for a in authors_str.split(',')]
        else:
            authors = []

        # Publication date
        pub_m = re.search(r'<pubDate>(.*?)</pubDate>', item)
        if pub_m:
            try:
                pub_date = parsedate_to_datetime(pub_m.group(1).strip())
                published = pub_date.strftime('%Y-%m-%d')
            except Exception:
                published = datetime.now().strftime('%Y-%m-%d')
        else:
            published = datetime.now().strftime('%Y-%m-%d')

        # Categories
        categories = re.findall(r'<category>(.*?)</category>', item)
        primary_category = categories[0] if categories else category

        papers.append({
            'id': paper_id,
            'title': title,
            'summary': summary,
            'authors': authors[:5],
            'published': published,
            'url': f'https://arxiv.org/abs/{paper_id}',
            'pdf_url': f'https://arxiv.org/pdf/{paper_id}.pdf',
            'category': primary_category,
        })

    return papers


def fetch_papers_for_channel(channel_name, config):
    """Fetch latest papers for a specific channel using RSS feeds.

    RSS feeds return all papers from the latest arXiv update (daily).
    No rate limiting, no date filtering needed (RSS only has today's papers).
    """
    channel = config["channels"][channel_name]
    categories = channel["categories"]
    keywords = [kw.lower() for kw in channel["keywords"]]
    min_score = config.get("min_score", 3)
    limit = config.get("per_channel_limit", 5)

    # Date filter: keep today + yesterday (covers timezone differences)
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    valid_dates = {today.isoformat(), yesterday.isoformat()}

    papers = []

    for i, category in enumerate(categories):
        if i > 0:
            time.sleep(1)  # Small delay between RSS requests

        data = fetch_rss_category(category)
        if not data:
            continue

        rss_papers = parse_rss_items(data, category)

        for p in rss_papers:
            # Date filter
            if p['published'] not in valid_dates:
                continue

            # Keyword scoring
            text = (p['title'] + ' ' + p['summary']).lower()
            score = 0
            for kw in keywords:
                if ' ' in kw or len(kw) <= 2:
                    if kw in text:
                        score += 1
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text):
                        score += 1

            if score >= min_score:
                p['channel'] = channel_name
                p['channel_label'] = channel["label"]
                p['relevance_score'] = score
                papers.append(p)

    papers.sort(key=lambda x: (-x['relevance_score'], x['published']))
    seen = set()
    unique = []
    for p in papers:
        if p['id'] not in seen:
            seen.add(p['id'])
            unique.append(p)

    return unique[:limit]


def main():
    parser = argparse.ArgumentParser(description='Fetch latest AI research papers from arXiv by channel')
    parser.add_argument('--channel', type=str, required=True, help='Channel name (infra/algorithms/architecture)')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--download', action='store_true', help='Download PDF files')

    args = parser.parse_args()

    config = load_config()

    if args.channel not in config["channels"]:
        print(f"Error: channel '{args.channel}' not found. Available: {list(config['channels'].keys())}")
        sys.exit(1)

    papers = fetch_papers_for_channel(args.channel, config)

    papers_dir = Path(__file__).resolve().parent.parent / "pdfs"
    pdfs_downloaded = []

    if args.download and papers:
        papers_dir.mkdir(parents=True, exist_ok=True)
        for p in papers:
            pdf_path = papers_dir / f"{p['id']}.pdf"
            if pdf_path.exists():
                pdfs_downloaded.append(str(pdf_path))
                continue
            try:
                subprocess.run(['curl', '-sL', '--max-time', '60', p['pdf_url'], '-o', str(pdf_path)], timeout=70)
                if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                    pdfs_downloaded.append(str(pdf_path))
            except Exception:
                pass

    result = {
        'channel': args.channel,
        'papers': papers,
        'total': len(papers),
        'fetched_at': datetime.utcnow().isoformat() + 'Z',
        'pdfs_downloaded': pdfs_downloaded
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Channel: {args.channel} - Fetched {len(papers)} papers")

    return result


if __name__ == '__main__':
    main()
