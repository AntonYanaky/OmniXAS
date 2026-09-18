[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Destination = (Join-Path $HOME "OmniXAS_data\figshare_kedge_xanes")
)

$ErrorActionPreference = "Stop"
$Url = "https://ndownloader.figshare.com/files/9932248"
$ApiUrl = "https://api.figshare.com/v2/articles/5678998"
$ArchiveName = "xas.json.tgz"
$ExpectedMd5 = "e866677ebb9270aeb2e15c725bef7e05"

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if ($null -eq $curl) {
    throw "curl.exe is required. Install curl or use a current Windows release."
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$MetadataPath = Join-Path $Destination "article_5678998.json"
$MetadataTempPath = "$MetadataPath.tmp"
$ArchivePath = Join-Path $Destination $ArchiveName
$PartialPath = "$ArchivePath.part"

Write-Host "Downloading Figshare article metadata to $MetadataPath"
& $curl.Source -L --fail --retry 5 -o $MetadataTempPath $ApiUrl
if ($LASTEXITCODE -ne 0) {
    Remove-Item -Force -ErrorAction SilentlyContinue $MetadataTempPath
    throw "The metadata download failed with exit code $LASTEXITCODE."
}
Move-Item -Force $MetadataTempPath $MetadataPath

$ArchiveIsValid = $false
if (Test-Path $ArchivePath) {
    $ArchiveIsValid = (Get-FileHash -Algorithm MD5 -Path $ArchivePath).Hash.ToLowerInvariant() -eq $ExpectedMd5
}

if ($ArchiveIsValid) {
    Write-Host "The verified archive already exists at $ArchivePath"
}
else {
    if (Test-Path $ArchivePath) {
        Write-Host "The existing archive failed verification; replacing it safely"
        Remove-Item -Force $ArchivePath
    }
    # Start from zero: blindly resuming is unsafe when a server ignores Range.
    Remove-Item -Force -ErrorAction SilentlyContinue $PartialPath
    Write-Host "Downloading the 5.56 GB archive to $PartialPath"
    & $curl.Source -L --fail --retry 5 -o $PartialPath $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed. Remove $PartialPath and run this script again."
    }

    Write-Host "Checking archive MD5"
    $ActualMd5 = (Get-FileHash -Algorithm MD5 -Path $PartialPath).Hash.ToLowerInvariant()
    if ($ActualMd5 -ne $ExpectedMd5) {
        throw "MD5 mismatch. Expected $ExpectedMd5 but got $ActualMd5. Remove $PartialPath and run this script again."
    }
    Move-Item -Force $PartialPath $ArchivePath
}

Write-Host "Download complete. Keep the archive compressed."
Write-Host "The Python extractor reads $ArchivePath directly."
