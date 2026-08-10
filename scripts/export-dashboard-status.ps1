[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('groww', 'alpaca', 'ccxt')]
    [string]$Broker,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Path
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
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

Push-Location -LiteralPath $repositoryRoot
try {
    & python -m market_sentinel.cli export-dashboard --broker $Broker --path $destination
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
