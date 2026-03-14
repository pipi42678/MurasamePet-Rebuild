Param (
    [ValidateSet("CU126", "CU128", "CPU")]
    [string]$Device = "CPU",
    [Parameter(Mandatory = $true)]
    [ValidateSet("HF", "HF-Mirror", "ModelScope")]
    [string]$Source,
    [switch]$DownloadUVR5
)

$ErrorActionPreference = 'Stop'

function Write-Log { param([string]$Message, [string]$Color = "White") Write-Host "[LOG]: $Message" -ForegroundColor $Color }

function Invoke-Download {
    param([string]$Uri, [string]$OutFile)
    Write-Log "Downloading: $OutFile" "Green"
    Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing
}

function Invoke-Unzip {
    param([string]$ZipPath, [string]$Destination)
    if (-not (Test-Path $Destination)) { New-Item -ItemType Directory -Path $Destination -Force | Out-Null }
    Write-Log "Extracting to: $Destination" "Cyan"
    Expand-Archive -Path $ZipPath -DestinationPath $Destination -Force
    Remove-Item $ZipPath -Force
}

Set-Location $PSScriptRoot

# --- Config Source ---
$Base = ""
if ($Source -eq "HF") { $Base = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
elseif ($Source -eq "HF-Mirror") { $Base = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
else { $Base = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master" }

# --- 1. Pretrained Models ---
if (-not (Test-Path "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained")) {
    Invoke-Download -Uri "$Base/pretrained_models.zip" -OutFile "pretrained_models.zip"
    Invoke-Unzip -ZipPath "pretrained_models.zip" -Destination "GPT_SoVITS"
}

# --- 2. G2PW Models ---
if (-not (Test-Path "GPT_SoVITS/text/G2PWModel")) {
    Invoke-Download -Uri "$Base/G2PWModel.zip" -OutFile "G2PWModel.zip"
    Invoke-Unzip -ZipPath "G2PWModel.zip" -Destination "GPT_SoVITS/text"
}

# --- 3. Python Environment Assets ---
$pythonCmd = (Get-Command python, python3 -ErrorAction SilentlyContinue | Select-Object -First 1).Source

if ($pythonCmd) {
    Write-Log "Using Python: $pythonCmd" "Cyan"
    $pythonPrefix = (& $pythonCmd "-c" "import sys; print(sys.prefix)").Trim()

    # NLTK
    if (-not (Test-Path (Join-Path $pythonPrefix "nltk_data"))) {
        Invoke-Download -Uri "$Base/nltk_data.zip" -OutFile "nltk_data.zip"
        Invoke-Unzip -ZipPath "nltk_data.zip" -Destination $pythonPrefix
    }

    # Open JTalk
    try {
        $pyopenjtalkPath = (& $pythonCmd "-c" "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
        if ($pyopenjtalkPath -and -not (Test-Path (Join-Path $pyopenjtalkPath "open_jtalk_dic_utf_8-1.11"))) {
            Invoke-Download -Uri "$Base/open_jtalk_dic_utf_8-1.11.tar.gz" -OutFile "oj.tar.gz"
            tar -xzf "oj.tar.gz" -C $pyopenjtalkPath
            Remove-Item "oj.tar.gz" -Force
        }
    } catch {
        Write-Log "Skipping OpenJTalk dict check." "Yellow"
    }
} else {
    Write-Log "Python not found. Skipping env assets." "Red"
}

Write-Log "All tasks completed successfully." "Green"