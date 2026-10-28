# Keeps Windows from sleeping while results_v3\QUEUE_RUNNING exists; exits afterwards.
# Uses SetThreadExecutionState only; no power settings are changed.
$flag = Join-Path $PSScriptRoot "..\..\results_v3\QUEUE_RUNNING"
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
Start-Sleep -Seconds 30
while (Test-Path $flag) {
    [Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null       # ES_CONTINUOUS: release
