param([Parameter(Mandatory=$true)][string]$InputPath,[Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference='Stop'
$word=$null
$document=$null
$existingCount=0
try {
    Write-Output 'Stage create application'
    $word=New-Object -ComObject Word.Application
    $existingCount=$word.Documents.Count
    $previousVisible=$word.Visible
    $previousAlerts=$word.DisplayAlerts
    $previousSecurity=$word.AutomationSecurity
    if ($existingCount -eq 0) { $word.Visible=$false }
    $word.DisplayAlerts=0
    $word.AutomationSecurity=3
    $confirm=$false
    $readOnly=$true
    $recent=$false
    Write-Output 'Stage open document'
    $document=$word.Documents.Open([ref]$InputPath,[ref]$confirm,[ref]$readOnly,[ref]$recent)
    Write-Output 'Stage export PDF'
    $document.ExportAsFixedFormat($OutputPath,17)
    Write-Output $OutputPath
} catch {
    Write-Output $_.ScriptStackTrace
    throw
} finally {
    $saveChanges=0
    if ($null -ne $document) { $document.Close([ref]$saveChanges); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($null -ne $word) {
        if ($existingCount -eq 0) { $word.Quit([ref]$saveChanges) }
        else { $word.Visible=$previousVisible; $word.DisplayAlerts=$previousAlerts; $word.AutomationSecurity=$previousSecurity }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
