function anom (args)
sc=subwrd(args,1)
if (sc<=0)
  sc = 0.5
endif
say 'set 'sc' anom colors'
c1  = -6.0*sc
c2  = -5.0*sc
c3  = -4.0*sc
c4  = -3.0*sc
c5  = -2.0*sc
c6  = -1.0*sc
c7  =  1.0*sc
c8  =  2.0*sc
c9  =  3.0*sc
c10 =  4.0*sc
c11 =  5.0*sc
c12 =  6.0*sc
'set rgb 21 128 0 0'
'set rgb 22 255 0 0'
'set rgb 24 255 128 0'
'set rgb 25 255 192 0'
'set rgb 41 191 191 191'
'set rgb 40 255 255 140'
'set rgb 26 128 255 128' 
'set rgb 29 0 255 255'
'set rgb 30 0 128 255'
'set rgb 31 192 255 160'
'set rgb 36 220 0 150'
'set gxout shaded'
'set clevs   'c1' 'c2' 'c3' 'c4' 'c5' 'c6' 'c7' 'c8' 'c9' 'c10' 'c11' 'c12
'set ccols   14  9   30   29   26   31   41   40   25   24   22   36   21'
