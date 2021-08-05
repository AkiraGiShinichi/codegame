@echo on
@REM set CMDER_ROOT=%~dp0
set CMDER_ROOT=D:\Windows\Programs\cmder
echo CMDER_ROOT
start %CMDER_ROOT%\vendor\conemu-maximus5\ConEmu.exe /icon "%CMDER_ROOT%\cmder.exe" /title Cmder /loadcfgfile "%CMDER_ROOT%\config\ConEmu.xml" /cmd cmd /k ""%CMDER_ROOT%\vendor\init.bat" && code ."