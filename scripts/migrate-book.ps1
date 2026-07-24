$ErrorActionPreference = 'Stop'

$Repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$SourceRoot = 'C:\Users\hw\Documents\modern-gpu-programming-for-mlsys'
$ZhRoot = Join-Path $SourceRoot 'zh'
$DocsRoot = Join-Path $Repo 'src\content\docs\books\modern-gpu-programming-for-mlsys'
$AssetRoot = Join-Path $Repo 'public\books\modern-gpu-programming-for-mlsys'
$BookBase = '/books/modern-gpu-programming-for-mlsys'

New-Item -ItemType Directory -Force -Path $DocsRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AssetRoot 'img') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $AssetRoot 'demo') | Out-Null

$Pages = @(
  @{ Src = 'index.md'; Out = 'index.md'; Order = 1; Title = 'Modern GPU Programming For MLSys' },
  @{ Src = 'chapter_background\index.md'; Out = 'gpu-execution-model.md'; Order = 10 },
  @{ Src = 'chapter_performance\index.md'; Out = 'performance.md'; Order = 20 },
  @{ Src = 'chapter_data_layout\index.md'; Out = 'data-layout.md'; Order = 30 },
  @{ Src = 'chapter_layout_generations\index.md'; Out = 'layout-generations.md'; Order = 40 },
  @{ Src = 'chapter_tma\index.md'; Out = 'tma.md'; Order = 50 },
  @{ Src = 'chapter_tensor_cores\index.md'; Out = 'tensor-cores.md'; Order = 60 },
  @{ Src = 'chapter_tmem\index.md'; Out = 'tmem.md'; Order = 70 },
  @{ Src = 'chapter_async_barriers\index.md'; Out = 'async-barriers.md'; Order = 80 },
  @{ Src = 'chapter_clc\index.md'; Out = 'cluster-launch-control.md'; Order = 90 },
  @{ Src = 'chapter_intro_tirx\index.md'; Out = 'tirx-intro.md'; Order = 100 },
  @{ Src = 'chapter_tirx_layout_api\index.md'; Out = 'tirx-layout-api.md'; Order = 110 },
  @{ Src = 'chapter_gemm_basics\index.md'; Out = 'gemm-basics.md'; Order = 120 },
  @{ Src = 'chapter_gemm_async\index.md'; Out = 'gemm-async.md'; Order = 130 },
  @{ Src = 'chapter_gemm_advanced\index.md'; Out = 'gemm-advanced.md'; Order = 140 },
  @{ Src = 'chapter_flash_attention\index.md'; Out = 'flash-attention.md'; Order = 150 },
  @{ Src = 'appendix\index.md'; Out = 'appendix.md'; Order = 160 },
  @{ Src = 'appendix\debugging_warp_specialized.md'; Out = 'debugging-warp-specialized.md'; Order = 170 }
)

$RefMap = @{
  'chap_background' = @{ Title = 'GPU execution model'; Slug = 'gpu-execution-model' }
  'chap_performance' = @{ Title = 'performance'; Slug = 'performance' }
  'chap_data_layout' = @{ Title = 'data layout'; Slug = 'data-layout' }
  'chap_layout_generations' = @{ Title = 'layout generations'; Slug = 'layout-generations' }
  'chap_tma' = @{ Title = 'TMA'; Slug = 'tma' }
  'chap_tensor_cores' = @{ Title = 'Tensor Core'; Slug = 'tensor-cores' }
  'chap_tmem' = @{ Title = 'TMEM'; Slug = 'tmem' }
  'chap_async_barriers' = @{ Title = 'mbarrier'; Slug = 'async-barriers' }
  'chap_clc' = @{ Title = 'cluster launch control'; Slug = 'cluster-launch-control' }
  'chap_intro_tirx' = @{ Title = 'TIRx intro'; Slug = 'tirx-intro' }
  'chap_tirx_layout_api' = @{ Title = 'TIRx layout API'; Slug = 'tirx-layout-api' }
  'chap_gemm_basics' = @{ Title = 'GEMM basics'; Slug = 'gemm-basics' }
  'chap_gemm_async' = @{ Title = 'GEMM async'; Slug = 'gemm-async' }
  'chap_gemm_advanced' = @{ Title = 'advanced GEMM'; Slug = 'gemm-advanced' }
  'chap_flash_attention' = @{ Title = 'Flash Attention 4'; Slug = 'flash-attention' }
  'chap_arch' = @{ Title = 'compiler internals'; Slug = 'appendix' }
  'chap_language_reference' = @{ Title = 'TIRx language reference'; Slug = 'appendix' }
  'chap_tirx_primer' = @{ Title = 'TIRx intro'; Slug = 'tirx-intro' }
  'chap_warp_spec_debug' = @{ Title = 'debugging warp specialization'; Slug = 'debugging-warp-specialized' }
  'chap_cta_cluster' = @{ Title = 'CTA cluster'; Slug = 'gemm-advanced' }
  'chap_multi_consumer' = @{ Title = 'multi-consumer GEMM'; Slug = 'gemm-advanced' }
  'chap_warp_specialization' = @{ Title = 'warp specialization'; Slug = 'gemm-advanced' }
}

function Convert-BookMarkdown {
  param([string]$Text)

  $Text = $Text -replace '(?m)^\([^\r\n]+\)=\r?\n', ''
  $Text = $Text -replace '(?s)```\{toctree\}.*?```', ''
  $Text = [regex]::Replace($Text, '(?ms)^```\{raw\} html\s*\r?\n(.*?)^```\s*$', '$1')
  $Text = $Text -replace '(?m)^#\s+.+\r?\n', ''
  $Text = $Text -replace '(?m)^:::\{admonition\} ([^\r\n]+)\r?\n', (':::note[$1]' + "`n")
  $Text = $Text -replace '(?m)^::::\{admonition\} ([^\r\n]+)\r?\n', (':::note[$1]' + "`n")
  $Text = $Text -replace '(?m)^:class: [^\r\n]+\r?\n', ''
  $Text = $Text -replace '(?m)^::::\s*$', ':::'
  $Text = [regex]::Replace($Text, 'style="width:100%;\s*min-width:\s*(\d+px);\s*([^"]*)"', {
    param($Match)
    $Width = $Match.Groups[1].Value
    $Rest = $Match.Groups[2].Value
    return 'style="width:' + $Width + '; max-width:none; ' + $Rest + '"'
  })
  $Text = [regex]::Replace($Text, '(<iframe\b[^>]*\bsrc="[^"?]+\.html)(")', {
    param($Match)
    return $Match.Groups[1].Value + '?notitle' + $Match.Groups[2].Value
  })
  $Text = [regex]::Replace($Text, '(?s)(<iframe\b(?=.*?style="width:\d+px; max-width:none;).*?</iframe>)', {
    param($Match)
    return '<div style="overflow-x:auto;">' + "`n" + $Match.Groups[1].Value + "`n" + '</div>'
  })
  $Text = $Text -replace '\.\./img/', "$BookBase/img/"
  $Text = $Text -replace '\.\./demo/', "$BookBase/demo/"
  $Text = $Text -replace '\.\./_static/tirx-layout-demo/', "$BookBase/_static/tirx-layout-demo/"

  $Text = [regex]::Replace($Text, '\{ref\}`([^`<]+?)\s*<([^>]+)>`', {
    param($Match)
    $Caption = $Match.Groups[1].Value.Trim()
    $Label = $Match.Groups[2].Value.Trim()
    if ($RefMap.ContainsKey($Label)) {
      $Slug = $RefMap[$Label].Slug
      return '[' + $Caption + '](' + $BookBase + '/' + $Slug + '/)'
    }
    return $Caption
  })

  foreach ($Label in $RefMap.Keys) {
    $Title = $RefMap[$Label].Title
    $Slug = $RefMap[$Label].Slug
    $Pattern = [regex]::Escape('{ref}`' + $Label + '`')
    $Replacement = '[' + $Title + '](' + $BookBase + '/' + $Slug + '/)'
    $Text = [regex]::Replace($Text, $Pattern, $Replacement)
  }

  return $Text.Trim() + "`n"
}

foreach ($Page in $Pages) {
  $Source = Join-Path $ZhRoot $Page.Src
  $Target = Join-Path $DocsRoot $Page.Out
  $Text = Get-Content -Raw -Encoding UTF8 -LiteralPath $Source
  $Title = $Page.Title

  if (-not $Title) {
    $Match = [regex]::Match($Text, '(?m)^#\s+(.+?)\s*$')
    if ($Match.Success) {
      $Title = $Match.Groups[1].Value.Trim()
    } else {
      $Title = [IO.Path]::GetFileNameWithoutExtension($Page.Out)
    }
  }

  $Body = Convert-BookMarkdown $Text
  $Frontmatter = "---`ntitle: $Title`nsidebar:`n  order: $($Page.Order)`n---`n`n"
  Set-Content -Encoding UTF8 -LiteralPath $Target -Value ($Frontmatter + $Body)
}

Get-ChildItem -LiteralPath (Join-Path $SourceRoot '_images') | Copy-Item -Destination (Join-Path $AssetRoot 'img') -Recurse -Force
Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'demo') | Copy-Item -Destination (Join-Path $AssetRoot 'demo') -Recurse -Force

# viz-base.js / viz-base.css sit at the docs_build/site root and are referenced by every
# demo HTML via ../viz-base.{js,css}. Without them every iframe renders blank (404).
Copy-Item -LiteralPath (Join-Path $SourceRoot 'docs_build\site\viz-base.js') -Destination (Join-Path $AssetRoot 'viz-base.js') -Force
Copy-Item -LiteralPath (Join-Path $SourceRoot 'docs_build\site\viz-base.css') -Destination (Join-Path $AssetRoot 'viz-base.css') -Force

$TirxDemoTarget = Join-Path $AssetRoot '_static\tirx-layout-demo'
New-Item -ItemType Directory -Force -Path $TirxDemoTarget | Out-Null
Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'docs_build\site\_static\tirx-layout-demo') | Copy-Item -Destination $TirxDemoTarget -Recurse -Force
