<#
.SYNOPSIS
Exports a local redacted dashboard snapshot.

.NOTES
Windows supports handle-bound replacement of an existing destination. On POSIX,
use a fresh unique/versioned .json path for every export: safe mode atomically
creates an absent destination and refuses to replace an existing path.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('groww', 'alpaca', 'ccxt')]
    [string]$Broker,

    [Parameter(Mandatory = $true, HelpMessage = 'Use a fresh unique/versioned .json path on POSIX.')]
    [ValidateNotNullOrEmpty()]
    [string]$Path
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($Path.StartsWith('\\') -or $Path.StartsWith('//')) {
    throw 'Network dashboard paths are not permitted.'
}
if ([System.IO.Path]::IsPathRooted($Path)) {
    $destination = [System.IO.Path]::GetFullPath($Path)
}
else {
    $destination = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Path))
    $repositoryPrefix = $repositoryRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $destination.StartsWith($repositoryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Relative dashboard path must remain under the repository.'
    }
}
if ([System.IO.Path]::GetExtension($destination) -ne '.json') {
    throw 'Dashboard destination must use the .json extension.'
}

function Assert-LocalPathComponents {
    param([Parameter(Mandatory = $true)][string]$Candidate)
    if ($Candidate.StartsWith('\\') -or $Candidate.StartsWith('//')) {
        throw 'Network dashboard paths are not permitted.'
    }
    $root = [System.IO.Path]::GetPathRoot($Candidate)
    if ([string]::IsNullOrEmpty($root)) {
        throw 'Dashboard path root is invalid.'
    }
    try {
        $drive = [System.IO.DriveInfo]::new($root)
        if ($drive.DriveType -eq [System.IO.DriveType]::Network) {
            throw 'Network dashboard paths are not permitted.'
        }
    }
    catch {
        throw 'Dashboard path root is unavailable or unsafe.'
    }
    $current = $root
    $relative = $Candidate.Substring($root.Length)
    foreach ($component in $relative.Split([System.IO.Path]::DirectorySeparatorChar, [System.StringSplitOptions]::RemoveEmptyEntries)) {
        $current = Join-Path $current $component
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Dashboard path cannot contain a link or reparse point.'
            }
        }
    }
}

Assert-LocalPathComponents -Candidate $destination
$arguments = @(
    '-m',
    'market_sentinel.cli',
    'export-dashboard',
    '--broker',
    $Broker,
    '--path',
    $destination
)
$exitCode = 1

Push-Location -LiteralPath $repositoryRoot
try {
    & python @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
