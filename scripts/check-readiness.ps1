[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('groww', 'alpaca', 'ccxt')]
    [string]$Broker
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $repositoryRoot
try {
    & python -m market_sentinel.cli live-preflight --broker $Broker
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
