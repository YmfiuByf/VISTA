$ErrorActionPreference = "Continue"

$ProjectRoot = "E:\BaiduNetdiskDownload\DA\Yao Sun\VISTA"

function Ensure-Dir {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Move-IfExists {
    param(
        [string]$Source,
        [string]$DestinationDir
    )
    if (Test-Path $Source) {
        Ensure-Dir $DestinationDir
        Move-Item $Source $DestinationDir -Force
        Write-Host "Moved: $Source -> $DestinationDir"
    }
}

function Move-FilesByPattern {
    param(
        [string]$SourceDir,
        [string]$Pattern,
        [string]$DestinationDir
    )
    if (Test-Path $SourceDir) {
        Ensure-Dir $DestinationDir
        Get-ChildItem $SourceDir -File -Filter $Pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Move-Item $_.FullName $DestinationDir -Force
            Write-Host "Moved: $($_.FullName) -> $DestinationDir"
        }
    }
}

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item $Path -Recurse -Force
        Write-Host "Removed: $Path"
    }
}

function Remove-PyCacheRecursively {
    param([string]$Root)
    if (Test-Path $Root) {
        Get-ChildItem $Root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "__pycache__" } |
        ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force
            Write-Host "Removed cache: $($_.FullName)"
        }
    }
}

function Remove-EmptyDirs {
    param([string[]]$Roots)

    foreach ($root in $Roots) {
        if (Test-Path $root) {
            Get-ChildItem $root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object {
                $items = Get-ChildItem $_.FullName -Force -ErrorAction SilentlyContinue
                if ($null -eq $items -or $items.Count -eq 0) {
                    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                    Write-Host "Removed empty dir: $($_.FullName)"
                }
            }
        }
    }
}

Set-Location $ProjectRoot

# =========================
# 1) Dynamic_JSCC-main
# =========================
$dynRoot = Join-Path $ProjectRoot "Dynamic_JSCC-main"

if (Test-Path $dynRoot) {
    $dynExamples = Join-Path $dynRoot "examples"
    $dynInput = Join-Path $dynExamples "input"
    $dynOut = Join-Path $dynExamples "output\jscc"
    $dynFigures = Join-Path $dynExamples "figures"
    $dynScriptsExp = Join-Path $dynRoot "scripts\experiments"
    $dynLogs = Join-Path $dynRoot "logs"

    Ensure-Dir $dynInput
    Ensure-Dir $dynOut
    Ensure-Dir $dynFigures
    Ensure-Dir $dynScriptsExp
    Ensure-Dir $dynLogs

    Get-ChildItem $dynRoot -File -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name

        if ($name -eq "README.md") {
        }
        elseif ($name -eq "geckodriver.log") {
            Move-Item $_.FullName $dynLogs -Force
            Write-Host "Moved: $($_.FullName) -> $dynLogs"
        }
        elseif ($name -match "_JSCC") {
            Move-Item $_.FullName $dynOut -Force
            Write-Host "Moved: $($_.FullName) -> $dynOut"
        }
        elseif ($name -match "dyna_structure\.png" -or $name -match "\.jpg$" -or $name -match "\.png$") {
            if ($name -match "dyna_structure\.png") {
                Move-Item $_.FullName $dynFigures -Force
                Write-Host "Moved: $($_.FullName) -> $dynFigures"
            }
            else {
                Move-Item $_.FullName $dynInput -Force
                Write-Host "Moved: $($_.FullName) -> $dynInput"
            }
        }
        elseif ($name -match "\.py$") {
            Move-Item $_.FullName $dynScriptsExp -Force
            Write-Host "Moved: $($_.FullName) -> $dynScriptsExp"
        }
    }

    Remove-IfExists (Join-Path $dynRoot ".idea")
    Remove-IfExists (Join-Path $dynRoot ".pytest_cache")
    Remove-PyCacheRecursively $dynRoot
}

# =========================
# 2) mmsegmentation-master
# =========================
$mmRoot = Join-Path $ProjectRoot "mmsegmentation-master"

if (Test-Path $mmRoot) {
    $mmDemo = Join-Path $mmRoot "demo"
    $mmDemoScripts = Join-Path $mmDemo "scripts"
    $mmDemoNotebooks = Join-Path $mmDemo "notebooks"
    $mmDemoContours = Join-Path $mmDemo "images\contours"
    $mmDemoOutputs = Join-Path $mmDemo "images\outputs"
    $mmDemoTemp = Join-Path $mmDemo "images\temp"

    Ensure-Dir $mmDemoScripts
    Ensure-Dir $mmDemoNotebooks
    Ensure-Dir $mmDemoContours
    Ensure-Dir $mmDemoOutputs
    Ensure-Dir $mmDemoTemp

    if (Test-Path $mmDemo) {
        Get-ChildItem $mmDemo -File -ErrorAction SilentlyContinue | ForEach-Object {
            $name = $_.Name

            if ($name -match "\.ipynb$") {
                Move-Item $_.FullName $mmDemoNotebooks -Force
                Write-Host "Moved: $($_.FullName) -> $mmDemoNotebooks"
            }
            elseif ($name -match "^Contour") {
                Move-Item $_.FullName $mmDemoContours -Force
                Write-Host "Moved: $($_.FullName) -> $mmDemoContours"
            }
            elseif ($name -match "^tmp" -or $name -match "^out" -or $name -match "^demo\d*\.png$" -or $name -eq "segmentation.png" -or $name -eq "figure.png") {
                Move-Item $_.FullName $mmDemoOutputs -Force
                Write-Host "Moved: $($_.FullName) -> $mmDemoOutputs"
            }
            elseif ($name -match "\.py$") {
                Move-Item $_.FullName $mmDemoScripts -Force
                Write-Host "Moved: $($_.FullName) -> $mmDemoScripts"
            }
            elseif ($name -match "\.png$" -or $name -match "\.jpg$") {
                Move-Item $_.FullName $mmDemoTemp -Force
                Write-Host "Moved: $($_.FullName) -> $mmDemoTemp"
            }
        }
    }

    Remove-IfExists (Join-Path $mmDemo ".idea")
    Remove-PyCacheRecursively $mmRoot
}

# =========================
# 3) VFIformer-main
# =========================
$vfiRoot = Join-Path $ProjectRoot "VFIformer-main"

if (Test-Path $vfiRoot) {
    $vfiScripts = Join-Path $vfiRoot "scripts"
    $vfiScriptsExp = Join-Path $vfiScripts "experiments"
    $vfiScriptsDemo = Join-Path $vfiScripts "demo"
    $vfiScriptsTrain = Join-Path $vfiScripts "train"
    $vfiRuntime = Join-Path $vfiRoot "runtime"

    Ensure-Dir $vfiScriptsExp
    Ensure-Dir $vfiScriptsDemo
    Ensure-Dir $vfiScriptsTrain
    Ensure-Dir $vfiRuntime

    Get-ChildItem $vfiRoot -File -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name

        if ($name -eq "README.md") {
        }
        elseif ($name -match "train\.py") {
            Move-Item $_.FullName $vfiScriptsTrain -Force
            Write-Host "Moved: $($_.FullName) -> $vfiScriptsTrain"
        }
        elseif ($name -match "demo\.py" -or $name -match "FILM_test\.py") {
            Move-Item $_.FullName $vfiScriptsDemo -Force
            Write-Host "Moved: $($_.FullName) -> $vfiScriptsDemo"
        }
        elseif ($name -match "\.dll$") {
            Move-Item $_.FullName $vfiRuntime -Force
            Write-Host "Moved: $($_.FullName) -> $vfiRuntime"
        }
        elseif ($name -match "\.py$") {
            Move-Item $_.FullName $vfiScriptsExp -Force
            Write-Host "Moved: $($_.FullName) -> $vfiScriptsExp"
        }
    }

    Remove-PyCacheRecursively $vfiRoot
}

# =========================
# 4) videos
# =========================
$videosRoot = Join-Path $ProjectRoot "videos"

if (Test-Path $videosRoot) {
    $videoArchives = Join-Path $videosRoot "archives"
    $videoSheets = Join-Path $videosRoot "spreadsheets"
    $videoMeta = Join-Path $videosRoot "metadata"
    $videoMediaAvi = Join-Path $videosRoot "source_media\avi"
    $videoMediaImg = Join-Path $videosRoot "source_media\images"
    $videoRuns = Join-Path $videosRoot "runs"

    Ensure-Dir $videoArchives
    Ensure-Dir $videoSheets
    Ensure-Dir $videoMeta
    Ensure-Dir $videoMediaAvi
    Ensure-Dir $videoMediaImg
    Ensure-Dir $videoRuns

    Move-IfExists (Join-Path $videosRoot "final_street") $videoRuns
    Move-IfExists (Join-Path $videosRoot "final_street2") $videoRuns
    Move-IfExists (Join-Path $videosRoot "final_street3") $videoRuns
    Move-IfExists (Join-Path $videosRoot "final_street4") $videoRuns

    Get-ChildItem $videosRoot -File -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name

        if ($name -match "\.zip$") {
            Move-Item $_.FullName $videoArchives -Force
            Write-Host "Moved: $($_.FullName) -> $videoArchives"
        }
        elseif ($name -match "\.xlsx$") {
            Move-Item $_.FullName $videoSheets -Force
            Write-Host "Moved: $($_.FullName) -> $videoSheets"
        }
        elseif ($name -match "\.csv$" -or $name -match "\.txt$" -or $name -match "\.npy$") {
            Move-Item $_.FullName $videoMeta -Force
            Write-Host "Moved: $($_.FullName) -> $videoMeta"
        }
        elseif ($name -match "\.avi$") {
            Move-Item $_.FullName $videoMediaAvi -Force
            Write-Host "Moved: $($_.FullName) -> $videoMediaAvi"
        }
        elseif ($name -match "\.jpg$" -or $name -match "\.png$") {
            Move-Item $_.FullName $videoMediaImg -Force
            Write-Host "Moved: $($_.FullName) -> $videoMediaImg"
        }
    }
}

# =========================
# 5) root cleanup
# =========================
Remove-PyCacheRecursively $ProjectRoot

$rootsToClean = @(
    (Join-Path $ProjectRoot "Dynamic_JSCC-main"),
    (Join-Path $ProjectRoot "mmsegmentation-master"),
    (Join-Path $ProjectRoot "VFIformer-main"),
    (Join-Path $ProjectRoot "videos")
)

Remove-EmptyDirs $rootsToClean

Write-Host ""
Write-Host "Done."