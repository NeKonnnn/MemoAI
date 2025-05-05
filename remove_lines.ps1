$from = 1646
$to = 1736
$content = Get-Content -Path gui.py
$newContent = @()
for ($i=0; $i -lt $content.Length; $i++) {
    if ($i -lt ($from-1) -or $i -gt ($to-1)) {
        $newContent += $content[$i]
    }
}
$newContent | Set-Content -Path gui.py 