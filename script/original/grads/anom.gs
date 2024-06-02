'reinit'
'run white.gs'

pull file1_ctl
pull file2_ctl
pull file_txt
pull run_type

'open 'file1_ctl
'open 'file2_ctl

'set mpdset mres'
qq = ""
'set t 0'
'set lon 20 380'
'set lat -80 80'

say 'Enter Date Desired (e.g., 11may2012)'
say 'if date = 0 stop' 

while (1)
res = read(file_txt)
rc = sublin(res,1)
  if (rc>0); break; endif;
  rec = sublin(res,2)
say rec
day = subwrd(rec,1)
say day
month = subwrd(rec,2)
say month
year = subwrd(rec,3)
say year
endwhile

'set time ' day month year

'q dims'
  temp = sublin(result,5)
  say temp
  ts = subwrd (temp,6)
  tt = subwrd (temp,9)
  td = substr (ts,4,12)
  say 'date to be displayed = 'td
  
  'enable print sat-'run_type'-anom-'td'.prt' 
    'set grads off'
    'set grid off'
    'set mpt * off'
    'set mpt 0 -1'
    'set mpt 1 -1'
    'set ylint 20'
    'set xlint 40'
    'set map 1 1 1'    
    'set gxout grfill'
    
    'run anomg 0.5'    
      'd sst.1 - clm.2'

      'draw title AVHRR - only'

      'run cbarn-new.gs 1.0 0 5.5 0.7'
      
      'set font 5'
      'set strsiz .26'
      'set string 1 tc 6'
      'draw string 5.5 7.8 Daily OISST Anomaly 'run_type': 'td 
   
  'set font 0'
  say 'print sat-'run_type'-anom-' td
  'print'
  'printim navy-anom-s-'td'.gif x420 y300  white'
  'printim navy-anom-b-'td'.gif x1000 y772  white'
  'disable print'
  quit
