[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('groww', 'alpaca', 'ccxt')]
    [string]$Broker
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$arguments = @('-m', 'market_sentinel.cli', 'live-preflight', '--broker', $Broker)
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
