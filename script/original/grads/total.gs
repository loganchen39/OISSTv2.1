'reinit'
'run white.gs'

pull file1_ctl
pull file_txt
pull run_type

'open 'file1_ctl

'set mpdset mres'
qq = ""
'set t 0'
'set lon 20 380'
'set lat -80 80'

say 'Enter Date Desired (e.g., 11may2006)'

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
  say 'date to plotted = 'td
  
  'enable print sat-'run_type'-total-'td'.prt'
  
    'set grads off'
    'set mpt * off'
    'set mpt 0 -1'
    'set mpt 1 -1'
    'set grid off'
    'set ylint 20'
    'set xlint 40'
    'set map 1 1 1'    
    'set gxout grfill'
    'run color17.gs'  
     
      'd sst.1'
      
      'set gxout contour'
      'set cint 5'
      'set ccolor 1'
      'd sst.1'
      'set clevs 28'
      'set ccolor 3'
      'd sst.1'
      
      'set clevs 0'
      'set ccolor 2'
      'd sst.1'
     
      'draw title AVHRR - only' 

      'run cbarn-new.gs 1.0 0 5.5 0.7'
            
      'set font 5'
      'set strsiz .26'
      'set string 1 tc 6'
      'draw string 5 7.8 Daily OISST 'run_type': 'td 
      
  'set font 0'
  say 'print sat-'run_type'-total-' td
  'print'
  'printim navy-sst-s-'td'.gif x420 y300  white'
  'printim navy-sst-b-'td'.gif x1000 y772  white'
  'disable print'
 quit  
