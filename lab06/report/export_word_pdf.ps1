param([Parameter(Mandatory=$true)][string]$Docx,
      [Parameter(Mandatory=$true)][string]$Pdf)
$ErrorActionPreference = 'Stop'
$wordApp = $null
$wordDoc = $null
$createdEmptyInstance = $false
try {
    $docxPath = (Resolve-Path -LiteralPath $Docx).Path
    $pdfPath = [IO.Path]::GetFullPath($Pdf)
    $wordApp = New-Object -ComObject Word.Application
    if ($wordApp.Documents.Count -ne 0) { throw 'Word returned an instance containing existing documents; refusing to alter it.' }
    $createdEmptyInstance = $true
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordDoc = $wordApp.Documents.Open($docxPath, $false, $true)
    $wordDoc.ExportAsFixedFormat($pdfPath, 17)
    @{docx=$docxPath;pdf=$pdfPath;pages=$wordDoc.ComputeStatistics(2)} | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $wordDoc) { $wordDoc.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc) }
    if ($null -ne $wordApp) {
        if ($createdEmptyInstance) { $wordApp.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp)
    }
}
