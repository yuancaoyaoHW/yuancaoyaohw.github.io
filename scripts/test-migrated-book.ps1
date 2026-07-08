$ErrorActionPreference = 'Stop'

$DocsRoot = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')) 'src\content\docs\books\modern-gpu-programming-for-mlsys'
$Failures = New-Object System.Collections.Generic.List[string]

foreach ($File in Get-ChildItem -File -LiteralPath $DocsRoot -Filter '*.md') {
  $Text = Get-Content -Raw -Encoding UTF8 -LiteralPath $File.FullName
  $FenceCount = ([regex]::Matches($Text, '(?m)^```')).Count

  if ($FenceCount % 2 -ne 0) {
    $Failures.Add("$($File.Name): unbalanced fenced code blocks")
  }

  if ($Text -match '\{ref\}`[^`]+`') {
    $Failures.Add("$($File.Name): unresolved MyST ref")
  }

  if ($Text -match '(?m)^:::\{admonition\}') {
    $Failures.Add("$($File.Name): unresolved MyST admonition")
  }

  if ($Text -match 'min-width:\s*\d+px') {
    $Failures.Add("$($File.Name): fixed min-width can cause page overflow")
  }

  $WithoutFrontmatter = $Text -replace '(?s)^---\s.*?---\s*', ''
  $FirstMeaningfulLine = ($WithoutFrontmatter -split "`r?`n" | Where-Object { $_.Trim() })[0]
  if ($FirstMeaningfulLine -match '^# ') {
    $Failures.Add("$($File.Name): first body line is duplicate H1")
  }
}

if ($Failures.Count -gt 0) {
  $Failures | ForEach-Object { Write-Output "FAIL: $_" }
  exit 1
}

Write-Output 'Migrated book checks passed.'
