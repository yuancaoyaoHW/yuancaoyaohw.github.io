#!/usr/bin/env python3
"""
Fetch latest papers from arXiv by channel configuration.
Usage:
  python fetch_papers.py --channel infra --json
  python fetch_papers.py --channel infra --download --json
"""

import json
import subprocess
import re
import argparse
import sys
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "channels.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_papers_for_channel(channel_name, config):
    """Fetch latest papers for a specific channel."""
    channel = config["channels"][channel_name]
    categories = channel["categories"]
    keywords = [kw.lower() for kw in channel["keywords"]]
    min_score = config.get("min_score", 3)
    limit = config.get("per_channel_limit", 5)

    papers = []

    for category in categories:
        url = f'https://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending'

        try:
            result = subprocess.run(['curl', '-sL', url], capture_output=True, timeout=30)
            data = result.stdout.decode('utf-8', errors='replace')
        except Exception:
            continue

        entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)

        for entry in entries:
            id_match = re.search(r'<id>(.*?)</id>', entry)
            if id_match:
                paper_id = id_match.group(1).split('/abs/')[-1].split('v')[0]
            else:
                continue

            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title.group(1).replace('\n', ' ').strip() if title else "No title"

            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            summary = summary.group(1).replace('\n', ' ').strip() if summary else ""

            authors = re.findall(r'<name>(.*?)</name>', entry)[:3]
            published = re.search(r'<published>(\d{4}-\d{2}-\d{2})</published>', entry)
            published = published.group(1) if published else datetime.now().strftime('%Y-%m-%d')

            text = (title + ' ' + summary).lower()
            score = sum(1 for kw in keywords if kw in text)

            if score >= min_score:
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'summary': summary,
                    'authors': authors,
                    'published': published,
                    'url': f'https://arxiv.org/abs/{paper_id}',
                    'pdf_url': f'https://arxiv.org/pdf/{paper_id}.pdf',
                    'category': category,
                    'channel': channel_name,
                    'channel_label': channel["label"],
                    'relevance_score': score
                })

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
                subprocess.run(['curl', '-sL', p['pdf_url'], '-o', str(pdf_path)], timeout=60)
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
