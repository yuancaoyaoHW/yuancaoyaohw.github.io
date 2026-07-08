$ErrorActionPreference = 'Stop'

$Repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$SourceRoot = 'C:\Users\hw\Documents\modern-gpu-programming-for-mlsys'
$ZhRoot = Join-Path $SourceRoot 'zh'
$DocsRoot = Join-Path $Repo 'src\content\docs\books\modern-gpu-programming-for-mlsys'
$AssetRoot = Join-Path $Repo 'public\books\modern-gpu-programming-for-mlsys'

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

function Convert-BookMarkdown {
  param([string]$Text)

  $Text = $Text -replace '(?m)^\([^\r\n]+\)=\r?\n', ''
  $Text = $Text -replace '(?s)```\{toctree\}.*?```', ''
  $Text = $Text -replace '(?m)^:::\{admonition\} ([^\r\n]+)\r?\n', "> **`$1**`n"
  $Text = $Text -replace '(?m)^::::\{admonition\} ([^\r\n]+)\r?\n', "> **`$1**`n"
  $Text = $Text -replace '(?m)^:class: [^\r\n]+\r?\n', ''
  $Text = $Text -replace '(?m)^:::\s*$', ''
  $Text = $Text -replace '(?m)^::::\s*$', ''
  $Text = $Text -replace '(?m)^```\{raw\} html\s*$', ''
  $Text = $Text -replace '(?m)^```\s*$', ''
  $Text = $Text -replace '\.\./img/', '/books/modern-gpu-programming-for-mlsys/img/'
  $Text = $Text -replace '\.\./demo/', '/books/modern-gpu-programming-for-mlsys/demo/'
  $Text = $Text -replace '\.\./_static/tirx-layout-demo/', '/books/modern-gpu-programming-for-mlsys/_static/tirx-layout-demo/'

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

$TirxDemoTarget = Join-Path $AssetRoot '_static\tirx-layout-demo'
New-Item -ItemType Directory -Force -Path $TirxDemoTarget | Out-Null
Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'docs_build\site\_static\tirx-layout-demo') | Copy-Item -Destination $TirxDemoTarget -Recurse -Force
