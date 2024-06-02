#!/usr/bin/env python

from .. import buildproperties as props
from datetime import datetime
import json
import sys
from config import satelliteTypes, oiProperties, common

# keys ignored during parse; not written out to script workfile
ignoredKeys = ('runTypeSatDataTypes')

def write_key_value(f, key, value):
  f.write(key + '=' +  str(value) + '\n')

def write_key_values(f, keyValueItems):  
  for key, value in keyValueItems:
    if key not in ignoredKeys:
      if type(value) is dict:
        write_key_values(f, value.iteritems())
      else:
        if type(value) is list:
          f.write(key + '=' + '(')
          
          for v in value:
            f.write('"' + v + '" ')
          
          f.write(')\n')
        else:
          f.write(key + '=' +  str(value) + '\n')

def main():
  curDate = datetime.strptime(sys.argv[2], '%Y%m%d').date()
  
  runTypeSatDataTypes = oiProperties['oiRunType'][str(sys.argv[1])]['runTypeSatDataTypes']
  satRunType = None
  
  for runTypeSatDataType in runTypeSatDataTypes:
    
    # check if run type sat type has both date fields defined
    if all(a in str(runTypeSatDataType.keys()) for a in ('useRangeStart', 'useRangeEnd')):
      
      # check if sat type is a defined sat. type
      if runTypeSatDataType['type'] in satelliteTypes.keys():
        
        if runTypeSatDataType['useRangeStart'] == '':
          print 'useRangeStart is empty; setting to curDate'
          rangeStartDate = curDate
        else:
          rangeStartDate = datetime.strptime(runTypeSatDataType['useRangeStart'], '%Y%m%d').date()
        
        if runTypeSatDataType['useRangeEnd'] == '':
          print 'useRangeEnd is empty; setting to curDate'
          rangeEndDate = curDate
        else:
          rangeEndDate = datetime.strptime(runTypeSatDataType['useRangeEnd'], '%Y%m%d').date()
        
        # when current date is in the satellite run type range, use it as the active type
        if rangeStartDate <= curDate and curDate <= rangeEndDate:
          print ('Run date ' + str(curDate) + ' between ' +
                 str(rangeStartDate) + ' and ' + str(rangeEndDate) +
                 ' using ' + runTypeSatDataType['type'])
          # stop if an item is found in the date range
          satRunType = satelliteTypes[runTypeSatDataType['type']]
          # add the run type key that'll be used as a lookup key
          satRunType['satRunType'] = runTypeSatDataType['type']
          break
      else:
        print 'invalid satellite type: ' + runTypeSatDataType['type']
    else:
      print 'satellite data must have both useRangeStart and useRangeEnd defined'
  
  # Satellite type was found with useable date range; write file
  if satRunType != None:
    f = open(props.BUILD + '/workfile', 'w')
    
    # write out build/project directories for the scripts
    write_key_value(f, 'DATA', props.DATA)
    write_key_value(f, 'TMP', props.TMP)
    write_key_value(f, 'SCRIPT', props.SCRIPT)
    write_key_value(f, 'BIN', props.BIN)
    write_key_value(f, 'GRADS', props.GRADS)
    write_key_value(f, 'GASCRP', props.GASCRP)
    write_key_value(f, 'GADDIR', props.GADDIR)
    write_key_value(f, 'WGRIB', props.WGRIB)
    
    # write out satellite specific properties
    write_key_values(f, satRunType.iteritems())
    
    # write out items used by all run types
    write_key_values(f, common.iteritems())
    
    # iterate over the key-value pairs using the run type arg from oisst.sh
    write_key_values(f, oiProperties['oiRunType'][str(sys.argv[1])].iteritems())
    
    f.close()
  else:
    sys.exit('ERROR: Could not find matching satellite type for ' + str(curDate))

if __name__ == '__main__':
    main()

