#!/usr/bin/env python

from .. import buildproperties as props
import json


#------------------------------------------------
#
# Start of OISST run properties
# info applies to s/w version 2.1
#
#------------------------------------------------

satelliteTypes = json.loads(json.dumps({

  #-------------------------------------------------
  #
  # Satellite Type Properties (satellite binary obs)
  #
  # - Structure:
  #    -- Each satellite file is identified by its date range
  #    -- Scripts currently supports 2 satellites: "satA" and "satB"
  #
  #    'name' : {
  #      'sat{A/B/etc}' : {
  #        'sat{A/B}Source' : int id,
  #        'sat{A/B}Files' : {
  #          '/path/to/data/file',
  #          '/path/to/a/second/file',
  #          '/path/to/others'
  #        }
  #      }
  #    }
  #
  #   -- The 'name' key must correspond to a key in the
  #      'SELECT CASE' statement within sat_navy_daily_firstguessclim.f90.
  #      This field is used in the control structure to determine
  #      the appropriate satellite file reader implementation.
  #
  #   -- Satellite data file types will be chosen
  #      based on how useRangeStart and useRangeEnd is defined within
  #      the interim/final oiRunType section.
  #
  #   -- If useRangeStart and useRangeEnd is empty then
  #      it will be handled as the current run day.
  #
  #   -- Files found in the satFiles object will be
  #      concatedated together during script execution.
  #      At least one valid filename is needed. satFiles' key's naming does
  #      not matter; satFiles' values are handled as an array.
  #
  # - Example: Two Satellite File Types
  # -----------------------------------
  #  'avhrr6hr' : {
  #    'satA' : {
  #      'satASource' : 8,
  #      'satAFiles' : [
  #        props.DATA + 'path/to/file1'
  #        props.DATA + 'path/to/file2'
  #        props.DATA + 'path/to/file3'
  #        props.DATA + 'path/to/file4'
  #      ]
  #    },
  #    'satB' : {
  #      'satBSource' : 12,
  #      'satBFiles' : [
  #        props.DATA + 'could/be/path/to/same/files/file1'
  #        props.DATA + 'could/be/path/to/same/files/file2'
  #        props.DATA + 'could/be/path/to/same/files/file3'
  #        props.DATA + 'could/be/path/to/same/files/file4'
  #      ]
  #    },
  #  },
  #
  #  'avhrr8day' : {
  #    'satA' : {
  #      'satASource' : 8,
  #      'satAFiles' : [
  #        # noaa-19 *.STND.NP.* files
  #        props.DATA + '/$curDateM8Year/input/satellite/avhrr/class/NPR.STND.NP.D$curDateM8Year2digit`expr $curDateM8JDay | cut -c1-2`?',
  #        props.DATA + '/$curYear/input/satellite/avhrr/class/NPR.STND.NP.D$curYear2digit`expr $curJulian | cut -c1-2`?',
  #        props.DATA + '/$curDateP8Year/input/satellite/avhrr/class/NPR.STND.NP.D$curDateP8Year2digit`expr $curDateP8JDay | cut -c1-2`?',
  #      ]
  #    },
  #    'satB' : {
  #      'satBSource' : 12,
  #      'satBFiles' : [
  #        # metop-2 *.STND.M2.* files
  #        props.DATA + '/$curDateM8Year/input/satellite/avhrr/class/NPR.STND.M2.D$curDateM8Year2digit`expr $curDateM8JDay | cut -c1-2`?',
  #        props.DATA + '/$curYear/input/satellite/avhrr/class/NPR.STND.M2.D$curYear2digit`expr $curJulian | cut -c1-2`?',
  #        props.DATA + '/$curDateP8Year/input/satellite/avhrr/class/NPR.STND.M2.D$curDateP8Year2digit`expr $curDateP8JDay | cut -c1-2`?'
  #      ]
  #    },
  #  }
  #
  #  -- Defined within interim/final 'oiRunType' --
  #
  #  'runTypeSatDataTypes' : [
  #    {
  #      'useRangeStart' : '20130501',
  #      'useRangeEnd' : '20130505',
  #      'type' : 'avhrr6hr'
  #    },
  #    {
  #      'useRangeStart' : '20130506',
  #      'useRangeEnd' : '20130515',
  #      'type' : 'avhrr8day'
  #    }
  #  ],
  #
  #  For a start and stop date range of 20130501 : 20130515
  #    6 hr config will be used 20130501 : 20130505
  #    8 day config will be used 20130506 : 20130515
  # -----------------------------------
  
  'avhrr6hr' : {
    'satA' : {
      'satASource' : 12,
      'satAFiles' : [
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curYear/mcsst_mta_d$curDate_s*',
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curDateM1Year/mcsst_mta_d$curDateM1_s2*',
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curDateP1Year/mcsst_mta_d$curDateP1_s0*'
      ]
    },
    'satB' : {
      'satBSource' : 11,
      'satBFiles' : [
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curYear/mcsst_mtb_d$curDate_s*',
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curDateM1Year/mcsst_mtb_d$curDateM1_s2*',
        props.DATA_PRELIM_INPUT + '/avhrr6hr/$curDateP1Year/mcsst_mtb_d$curDateP1_s0*'
      ]
    }

  },
   
 'avhrr8day' : {
    'satA' : {
      'satASource' : 12,
      'satAFiles' : [
        props.DATA_FINAL_INPUT + '/avhrr8day/$curYear12/mcsst_mta_d$curDate_s*',
        props.DATA_FINAL_INPUT + '/avhrr8day/$curDate12M1Year/mcsst_mta_d$curDateM1_s2*',
        props.DATA_FINAL_INPUT + '/avhrr8day/$curDate12P1Year/mcsst_mta_d$curDateP1_s0*'
      ]
    },
    'satB' : {
      'satBSource' : 11,
      'satBFiles' : [
        props.DATA_FINAL_INPUT + '/avhrr8day/$curYear12/mcsst_mtb_d$curDate_s*',
        props.DATA_FINAL_INPUT + '/avhrr8day/$curDate12M1Year/mcsst_mtb_d$curDateM1_s2*',
        props.DATA_FINAL_INPUT + '/avhrr8day/$curDate12P1Year/mcsst_mtb_d$curDateP1_s0*'
      ]
    }

  }

}))

oiProperties = json.loads(json.dumps({
  'oiRunType' : {
    
    #----------------------------------------------
    #
    # OI interim run properties 
    #
    #----------------------------------------------
    
    'interim' : {
    
      #--------------------------------------------
      #
      # define the satellite data file types and
      # date ranges (if used, leave empty otherwise)
      #
      #--------------------------------------------
      'runTypeSatDataTypes' : [
        {
          'useRangeStart' : '',
          'useRangeEnd'   : '',
          'type'          : 'avhrr6hr'
        }
      ],
    
      #--------------------------------------------
      #
      # interim settings
      #
      #--------------------------------------------
      
      # data prep and QC
      'covBuoy' : 0.1,
      'covArgo' : 0.1,
      'covShip' : 0.05,
      'covBuoyship' : 0.0,
      'covSat'  : 2.5,
      'covSat2' : 0.1,
      'covIce'  : 6.0,
      'covIce2' : 0.0,
      'iwget'   : 0,

      # eot weights
      'minz'  : 750,			# minimum number of smoothed zonal points
      'njj'   : 10,			# number of latitude points for zonal smoothing  
      'minob' : 1,			# minimum number of obs used in zonal average
      'ioff'  : 1,			# number of longitude boxes smoothed for zonal sub
      'joff'  : 1,			# number of latitude boxes smoothed for zonal sub
      'ndays' : 7,			# number of days used
      'drej'  : 5.0,			# data reject limit for absolute sst increment 
      'modes' : 130,			# number of modes used <131
      'nby'   : 7,			# number of buoy obs 
      'nag'   : 7,			# number of argo obs 
      'nsh'   : 1,			# number of ship obs
      'nst'   : 1,			# number of sat obs

      # 0 false, 1 true
      'useFutureSuperobs' : 0,          # EOT weighting option; will analysis window lookahead for data?

      # eot corrections
      'nuse' : 1,			# number of weights used (1, 3, or 5, default is 1) 
      'dfw1' : 1.0,			# factor to smooth the weights: range 0 to 1

      #OI
      'rmax'  : 400.0,			# maximum x or y distance (km) over which obs are used.
      'nmax'  : 22,			# maximum number of obs are used. Must be <201
      'ifil'  : 1,			# number of times 1-2-1 smoothing is done; <1 no smoothing 
      'rej'   : 5.0,			# bias reject limit for absolute super obs increment
      'nbias' : 4,			# number of files to read of EOT bias estimates
      'nsat'  : 1,			# number independent satellite bias data sets 
      'iwarn' : 20,			# Percentage of satellite data: if lower warning message printed

      'oiTitle' : '\'NOAA/NCEI 1/4 Degree Daily Optimum Interpolation Sea Surface Temperature (OISST) Analysis, Version 2.1 - Interim\'',
      
      #--------------------------------------------
      #
      # interim files
      #
      #--------------------------------------------
      #raw inputs
      'buoyshipSSTDaily' : props.DATA_PRELIM_INPUT + '/buoyship/argoNicoads/$curYear/mq.$curDate',
      'engIce'           : props.DATA_PRELIM_INPUT + '/ice/ncep/$curYear/eng.$curDate',
      'seaIce'           : props.DATA_PRELIM_INPUT + '/ice/ncep/$curYear/seaice.t00z.grb.grib2.$curDate',

      #static inputs
      'iceMask'   : props.DATA + '/common/static/ice_flags_mask.dat',
      'iceCoef'   : props.DATA + '/common/static/gsfc-fit-coef-fill-final',
      'iceFrzPnt' : props.DATA + '/common/fzsst/daily-fzsst',

      #obs inputs
      'buoyObsOut' : props.DATA_PRELIM_WORK + '/obs/buoyship/$curYear/buoy.$curDate',
      'argoObsOut' : props.DATA_PRELIM_WORK + '/obs/buoyship/$curYear/argo.$curDate',
      'shipObsOut' : props.DATA_PRELIM_WORK + '/obs/buoyship/$curYear/ship.$curDate',
      'iceGlobe'   : props.DATA_PRELIM_WORK + '/obs/ice/yyyy/ice.yyyyMMdd',

      # super obs out
      'buoySobsOut'      : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/buoy.$curDate',
      'argoSobsOut'      : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/argo.$curDate',
      'shipSobsOut'      : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/shipc.$curDate',
      'iceSobsOut'       : props.DATA_PRELIM_WORK + '/sobs/ice/$curYear/icesst.$curDate',

      'satADaySobsOut'   : props.DATA_PRELIM_WORK + '/sobs/metopa/$curYear/dsobs.$curDate',
      'satANightSobsOut' : props.DATA_PRELIM_WORK + '/sobs/metopa/$curYear/nsobs.$curDate',
      'satBDaySobsOut'   : props.DATA_PRELIM_WORK + '/sobs/metopb/$curYear/dsobs.$curDate',
      'satBNightSobsOut' : props.DATA_PRELIM_WORK + '/sobs/metopb/$curYear/nsobs.$curDate',

      # fg output file
      'EOTfg' : props.DATA_PRELIM_OUT + '/oiout/$fgYear/sst4-metopab-eot-intv2.$fgDate',

      # eot super obs date mask file/path format
      'buoySuperobsPathFormat'     : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/buoy.yyyyMMdd',
      'argoSuperobsPathFormat'     : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/argo.yyyyMMdd',

      # corrected ship superobs date mask file/path format
      'shipObsCorrectedPathFormat' : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/shipc.yyyyMMdd',

      # super obs
      'buoySuperobs'      : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/buoy.$curDate',
      'argoSuperobs'      : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/argo.$curDate',
      'shipObsCorrected'  : props.DATA_PRELIM_WORK + '/sobs/buoyship/$curYear/shipc.$curDate',
      'iceSuperobs'       : props.DATA_PRELIM_WORK + '/sobs/ice/$curYear/icesst.$curDate',

      'satADaySuperobs'   : props.DATA_PRELIM_WORK + '/sobs/metopa/$curYear/dsobs.$curDate',
      'satANightSuperobs' : props.DATA_PRELIM_WORK + '/sobs/metopa/$curYear/nsobs.$curDate',
      'satBDaySuperobs'   : props.DATA_PRELIM_WORK + '/sobs/metopb/$curYear/dsobs.$curDate',
      'satBNightSuperobs' : props.DATA_PRELIM_WORK + '/sobs/metopb/$curYear/nsobs.$curDate',

      # eot wt files
      'satADayWeights'   : props.DATA_PRELIM_WORK + '/eotwt/metopa/$curYear/dwt.$curDate',
      'satANightWeights' : props.DATA_PRELIM_WORK + '/eotwt/metopa/$curYear/nwt.$curDate',
      'satBDayWeights'   : props.DATA_PRELIM_WORK + '/eotwt/metopb/$curYear/dwt.$curDate',
      'satBNightWeights' : props.DATA_PRELIM_WORK + '/eotwt/metopb/$curYear/nwt.$curDate',

      #eot bias files
      'satADayAnalysis'   : props.DATA_PRELIM_WORK + '/eotbias/metopa/$curYear/dbias.$curDate',
      'satANightAnalysis' : props.DATA_PRELIM_WORK + '/eotbias/metopa/$curYear/nbias.$curDate',
      'satBDayAnalysis'   : props.DATA_PRELIM_WORK + '/eotbias/metopb/$curYear/dbias.$curDate',
      'satBNightAnalysis' : props.DATA_PRELIM_WORK + '/eotbias/metopb/$curYear/nbias.$curDate',

      # eot correction files
      'satADayCorrected'   : props.DATA_PRELIM_WORK + '/eotcor/metopa/$curYear/dsobsc.$curDate',
      'satANightCorrected' : props.DATA_PRELIM_WORK + '/eotcor/metopa/$curYear/nsobsc.$curDate',
      'satBDayCorrected'   : props.DATA_PRELIM_WORK + '/eotcor/metopb/$curYear/dsobsc.$curDate',
      'satBNightCorrected' : props.DATA_PRELIM_WORK + '/eotcor/metopb/$curYear/nsobsc.$curDate',
      
      # grid output files
      'buoyGridOut'      : props.DATA_PRELIM_WORK + '/grid/buoyship/$curYear/buoy.$curDate',
      'argoGridOut'      : props.DATA_PRELIM_WORK + '/grid/buoyship/$curYear/argo.$curDate',
      'shipGridOut'      : props.DATA_PRELIM_WORK + '/grid/buoyship/$curYear/ship.$curDate',      
      'iceCon720x360'    : props.DATA_PRELIM_WORK + '/grid/ice/con/yyyy/icecon720x360.yyyyMMdd',
      'iceCon1440x720'   : props.DATA_PRELIM_WORK + '/grid/ice/con/yyyy/icecon.yyyyMMdd',
      'iceCon'           : props.DATA_PRELIM_WORK + '/grid/ice/con/yyyy/icecon.yyyyMMdd',
      'iceCon7day'       : props.DATA_PRELIM_WORK + '/grid/ice/con/$curYear/icecon7.$curDate',
      'iceConMed'        : props.DATA_PRELIM_WORK + '/grid/ice/con-med/$curYear/icecon-med.$curDate',
      'iceSSTGrads'      : props.DATA_PRELIM_WORK + '/grid/ice/ice-sst/$curYear/icesst.$curDate',       

      'satADayGridOut'   : props.DATA_PRELIM_WORK + '/grid/metopa/$curYear/dgrid.sst.$curDate',
      'satANightGridOut' : props.DATA_PRELIM_WORK + '/grid/metopa/$curYear/ngrid.sst.$curDate', 
      'satBDayGridOut'   : props.DATA_PRELIM_WORK + '/grid/metopb/$curYear/dgrid.sst.$curDate',
      'satBNightGridOut' : props.DATA_PRELIM_WORK + '/grid/metopb/$curYear/ngrid.sst.$curDate',           

      # satellite tmp concat file
      'satSSTCatObs'     : props.DATA_PRELIM_WORK + '/obs/satsst/$curYear/satobs.$curJulian',    

      # OI
      'fgOisst' : props.DATA_PRELIM_OUT + '/oiout/$fgYear/sst4-metopab-eot-intv2.$fgDate',
      'oisst'   : props.DATA_PRELIM_OUT + '/oiout/$curYear/sst4-metopab-eot-intv2.$curDate',      

      # OUTPUT
      # checked in NetCDF processing F90
      'netCDFrunType' : '\'INTERIM\'',

      # grads runtype abbreviation
      'runTypeAbrv' : '\'intv2\'',

      # GrADS
      'gradsOutDir'     : props.DATA_PRELIM_OUT + '/map/$curYear',
      'gradsPubDir'     : props.DATA_PRELIM_OUT + '/map/published',
      'outputRunDate'   : props.TMP + '/output_run_date.txt',
      'TotalGS'         : props.GASCRP + '/total.gs',
      'AnomGS'          : props.GASCRP + '/anom.gs',
      'dailyCTL'        : props.TMP + '/intv2.ctl',
#     'dailyCTL'        : props.DATA + '/common/static/intv2.ctl',
      'frPathOnly'      : props.DATA_PRELIM_OUT + '/NetCDF/GHRSST/$curYear',
      'frPath'          : props.DATA_PRELIM_OUT + '/NetCDF/GHRSST/$curYear/FR-$curDate\'120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.xml\'',

      # NetCDF
      'nomadsCDFpathOnly'    : props.DATA_PRELIM_OUT + '/NetCDF/$curYear',
      'nomadsCDFpath'        : props.DATA_PRELIM_OUT + '/NetCDF/$curYear/oisst-avhrr-v02r01.$curDate\'_preliminary.nc\'',
      'nomadsIEEEpath'       : props.DATA_PRELIM_OUT + '/NetCDF/$curYear/oisst-avhrr-v02r01.$curDate\'_preliminary\'',
      'ghrsstCDFpathOnly'    : props.DATA_PRELIM_OUT + '/NetCDF/GHRSST/$curYear',
      'ghrsstCDFJPLpathOnly' : props.DATA_PRELIM_OUT + '/NetCDF/GHRSST/GDS2toNASA-JPL',
      'ghrsstCDFpath'        : props.DATA_PRELIM_OUT + '/NetCDF/GHRSST/$curYear/$curDate\'120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc\'',   
      'iceConMedNetcdf'      : props.DATA_PRELIM_WORK + '/grid/ice/con-med/$curYear/icecon-med.$curDate',
      'climpath_leap'        : props.DATA + '/common/clim/leap/clmoiv2.$curMonth$curDay',
      'climpath_noleap'      : props.DATA + '/common/clim/noleap/clmoiv2.$curMonth$curDay',
    },
      
    #----------------------------------------------
    #
    # OI final run properties 
    #
    #----------------------------------------------
      
    'final' : {
      
      #--------------------------------------------
      #
      # define the satellite data file types and
      # date ranges (if used, leave empty otherwise)
      #
      #--------------------------------------------
      'runTypeSatDataTypes' : [
        {
          'useRangeStart' : '',
          'useRangeEnd' : '',
          'type' : 'avhrr8day' 
        }
      ],
      
      #--------------------------------------------
      #
      # final settings
      #
      #--------------------------------------------
      
      # data prep and QC
      'covBuoy'     : 0.1,
      'covArgo'     : 0.1,
      'covShip'     : 0.05,
      'covBuoyship' : 0.0,
      'covSat'      : 2.5,
      'covSat2'     : 0.1,
      'covIce'      : 6.0,
      'covIce2'     : 0.0,
      'iwget'       : 0,
      
      # Initial data for EOT weights
      'minz'    : 750,				# minimum number of smoothed zonal points
      'njj'     : 10,				# number of latitude points for zonal smoothing  
      'minob'   : 1,				# minimum number of obs used in zonal average
      'ioff'    : 1,				# number of longitude boxes smoothed for zonal sub
      'joff'    : 1,				# number of latitude boxes smoothed for zonal sub
      'ndays'   : 15,				# number of days used
      'drej'    : 5.0,				# data REJECT LIMIT for ABSOLUTE SST increment 
      'modes'   : 130,				# number of modes used <131
      'nby'     : 7,				# number of buoy obs
      'nag'     : 7,				# number of argo obs
      'nsh'     : 1,				# number of ship obs
      'nst'     : 1,				# number of sat obs

      'yrbuoy'  : 15,				# character count of buoy first number of year
      'yrargo'  : 15,				# character count of argo first number of year
      'yrship'  : 15,				# character count of ship first number of year
      'yrsat'   : 15,				# character count of sat first number of year
      'daybuoy' : 40,				# character count of buoy first number of date
      'dayargo' : 40,				# character count of argo first number of date
      'dayship' : 40,				# character count of ship first number of date
      'daysat'  : 44,				# character count of sat first number of data

      # 0 false, 1 true
      'useFutureSuperobs' : 1,

      # Initial data for EOT corrections
      'nuse'  : 5,		# number of weights used
      'dfw1'  : 0.0625,		# factor to smooth the weights: range 0 to 1    
      'dfw2'  : 0.25,		# factor to smooth the weights: range 0 to 1
      'dfw3'  : 0.375,		# factor to smooth the weights: range 0 to 1
      'dfw4'  : 0.25,		# factor to smooth the weights: range 0 to 1
      'dfw5'  : 0.0625,		# factor to smooth the weights: range 0 to 1

      # OI
      'rmax'  : 400.0,		# maximum x or y distance (km) over which obs are used.
      'nmax'  : 22,		# maximum number of obs are used. Must be <201
      'ifil'  : 1,		# number of times 1-2-1 smoothing is done; <1 no smoothing 
      'rej'   : 5.0,		# bias reject limit for absolute super obs increment
      'nbias' : 4,		# number of files to read of EOT bias estimates
      'nsat'  : 1,		# number independent satellite bias data sets 
      'iwarn' : 20,		# Percentage of satellite data: if lower warning message printed

      'oiTitle' : '\'NOAA/NCEI 1/4 Degree Daily Optimum Interpolation Sea Surface Temperature (OISST) Analysis, Version 2.1 - Final\'',
        
      #--------------------------------------------
      #
      # final files
      #
      #--------------------------------------------
      
      #raw inputs
      'buoyshipSSTDaily' : props.DATA_FINAL_INPUT + '/buoyship/argoNicoads/$curYear12/mq.$curDate12',
      'engIce'           : props.DATA_FINAL_INPUT + '/ice/ncep/$curYear12/eng.$curDate12',
      'seaIce'           : props.DATA_FINAL_INPUT + '/ice/ncep/$curYear12/seaice.t00z.grb.grib2.$curDate12',

      #static inputs
      'iceMask'   : props.DATA + '/common/static/ice_flags_mask.dat',
      'iceCoef'   : props.DATA + '/common/static/gsfc-fit-coef-fill-final',
      'iceFrzPnt' : props.DATA + '/common/fzsst/daily-fzsst',

      #obs inputs
      'buoyObsOut'       : props.DATA_FINAL_WORK + '/obs/buoyship/$curYear12/buoy.$curDate12',
      'argoObsOut'       : props.DATA_FINAL_WORK + '/obs/buoyship/$curYear12/argo.$curDate12',
      'shipObsOut'       : props.DATA_FINAL_WORK + '/obs/buoyship/$curYear12/ship.$curDate12',
      'iceGlobe'         : props.DATA_FINAL_WORK + '/obs/ice/yyyy/ice.yyyyMMdd',

      # super obs
      'buoySobsOut'      : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear12/buoy.$curDate12',
      'argoSobsOut'      : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear12/argo.$curDate12',
      'shipSobsOut'      : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear12/shipc.$curDate12',
      'iceSobsOut'       : props.DATA_FINAL_WORK + '/sobs/ice/$curYear_ice/icesst.$curDate_ice',

      'satADaySobsOut'   : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear12/dsobs.$curDate12',
      'satANightSobsOut' : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear12/nsobs.$curDate12',
      'satBDaySobsOut'   : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear12/dsobs.$curDate12',
      'satBNightSobsOut' : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear12/nsobs.$curDate12',

      # fg output file
#     'EOTfg' : props.DATA_FINAL_OUT  + '/oiout/$fgYear12/sst4-metopab-eot-finv2.$fgDate12',
      'EOTfg' : props.DATA_PRELIM_OUT + '/oiout/$fgYear12/sst4-metopab-eot-intv2.$fgDate12',

      # eot super obs date mask file/path format
      'buoySuperobsPathFormat'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/buoy.yyyyMMdd',
      'argoSuperobsPathFormat'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/argo.yyyyMMdd',

      # corrected ship superobs date mask file/path format
      'shipObsCorrectedPathFormat' : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/shipc.yyyyMMdd',

      # oi super obs
      'buoySuperobs'       : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/buoy.$curDate',
      'argoSuperobs'       : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/argo.$curDate',
      'buoySuperobsM1'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateM1Year/buoy.$curDateM1',
      'buoySuperobsP1'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateP1Year/buoy.$curDateP1',
      'argoSuperobsM1'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateM1Year/argo.$curDateM1',
      'argoSuperobsP1'     : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateP1Year/argo.$curDateP1',

      # corrected ship bias
      'shipObsCorrected'   : props.DATA_FINAL_WORK + '/sobs/buoyship/$curYear/shipc.$curDate',
      'shipObsCorrectedM1' : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateM1Year/shipc.$curDateM1',
      'shipObsCorrectedP1' : props.DATA_FINAL_WORK + '/sobs/buoyship/$curDateP1Year/shipc.$curDateP1',

      'iceSuperobs'        : props.DATA_FINAL_WORK + '/sobs/ice/$curYear/icesst.$curDate',
      'iceSuperobsP1'      : props.DATA_FINAL_WORK + '/sobs/ice/$curDateP1Year/icesst.$curDateP1',
      'iceSuperobsM1'      : props.DATA_FINAL_WORK + '/sobs/ice/$curDateM1Year/icesst.$curDateM1',

      'satADaySuperobsPathFmt'   : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear/dsobs.yyyyMMdd',
      'satANightSuperobsPathFmt' : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear/nsobs.yyyyMMdd',
      'satBDaySuperobsPathFmt'   : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear/dsobs.yyyyMMdd',
      'satBNightSuperobsPathFmt' : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear/nsobs.yyyyMMdd',

      'satADaySuperobsP1'        : props.DATA_FINAL_WORK + '/sobs/metopa/$curDateP1Year/dsobs.$curDateP1',
      'satANightSuperobsP1'      : props.DATA_FINAL_WORK + '/sobs/metopa/$curDateP1Year/nsobs.$curDateP1',
      'satBDaySuperobsP1'        : props.DATA_FINAL_WORK + '/sobs/metopb/$curDateP1Year/dsobs.$curDateP1',
      'satBNightSuperobsP1'      : props.DATA_FINAL_WORK + '/sobs/metopb/$curDateP1Year/nsobs.$curDateP1',

      'satADaySuperobsM1'        : props.DATA_FINAL_WORK + '/sobs/metopa/$curDateM1Year/dsobs.$curDateM1',
      'satANightSuperobsM1'      : props.DATA_FINAL_WORK + '/sobs/metopa/$curDateM1Year/nsobs.$curDateM1',
      'satBDaySuperobsM1'        : props.DATA_FINAL_WORK + '/sobs/metopb/$curDateM1Year/dsobs.$curDateM1',
      'satBNightSuperobsM1'      : props.DATA_FINAL_WORK + '/sobs/metopb/$curDateM1Year/nsobs.$curDateM1',

      'satADaySuperobs'          : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear/dsobs.$curDate',
      'satANightSuperobs'        : props.DATA_FINAL_WORK + '/sobs/metopa/$curYear/nsobs.$curDate',
      'satBDaySuperobs'          : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear/dsobs.$curDate',
      'satBNightSuperobs'        : props.DATA_FINAL_WORK + '/sobs/metopb/$curYear/nsobs.$curDate',

      # EOT wt files
      # current run day
      'satADayWeights'     : props.DATA_FINAL_WORK + '/eotwt/metopa/$curYear/dwt.$curDate',
      'satANightWeights'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curYear/nwt.$curDate',
      'satBDayWeights'     : props.DATA_FINAL_WORK + '/eotwt/metopb/$curYear/dwt.$curDate',
      'satBNightWeights'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curYear/nwt.$curDate',

      # cur. day + 1
      'satADayWeightsP1'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP1Year/dwt.$curDateP1',
      'satANightWeightsP1' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP1Year/nwt.$curDateP1',
      'satBDayWeightsP1'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP1Year/dwt.$curDateP1',
      'satBNightWeightsP1' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP1Year/nwt.$curDateP1',

      # cur day + 2
      'satADayWeightsP2'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP2Year/dwt.$curDateP2',
      'satANightWeightsP2' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP2Year/nwt.$curDateP2',
      'satBDayWeightsP2'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP2Year/dwt.$curDateP2',
      'satBNightWeightsP2' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP2Year/nwt.$curDateP2',

      # cur day  + 3
      'satADayWeightsP3'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP3Year/dwt.$curDateP3',
      'satANightWeightsP3' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateP3Year/nwt.$curDateP3',
      'satBDayWeightsP3'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP3Year/dwt.$curDateP3',
      'satBNightWeightsP3' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateP3Year/nwt.$curDateP3',

      # cur day - 1
      'satADayWeightsM1'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM1Year/dwt.$curDateM1',
      'satANightWeightsM1' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM1Year/nwt.$curDateM1',
      'satBDayWeightsM1'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM1Year/dwt.$curDateM1',
      'satBNightWeightsM1' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM1Year/nwt.$curDateM1',

      # cur day - 2
      'satADayWeightsM2'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM2Year/dwt.$curDateM2',
      'satANightWeightsM2' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM2Year/nwt.$curDateM2',
      'satBDayWeightsM2'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM2Year/dwt.$curDateM2',
      'satBNightWeightsM2' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM2Year/nwt.$curDateM2',

      # cur day - 3
      'satADayWeightsM3'   : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM3Year/dwt.$curDateM3',
      'satANightWeightsM3' : props.DATA_FINAL_WORK + '/eotwt/metopa/$curDateM3Year/nwt.$curDateM3',
      'satBDayWeightsM3'   : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM3Year/dwt.$curDateM3',
      'satBNightWeightsM3' : props.DATA_FINAL_WORK + '/eotwt/metopb/$curDateM3Year/nwt.$curDateM3',

      # eot bias current day
      'satADayAnalysis'      : props.DATA_FINAL_WORK + '/eotbias/metopa/$curYear/dbias.$curDate',
      'satANightAnalysis'    : props.DATA_FINAL_WORK + '/eotbias/metopa/$curYear/nbias.$curDate',
      'satBDayAnalysis'      : props.DATA_FINAL_WORK + '/eotbias/metopb/$curYear/dbias.$curDate',
      'satBNightAnalysis'    : props.DATA_FINAL_WORK + '/eotbias/metopb/$curYear/nbias.$curDate',

      # eot bias current run day + 1
      'satADayAnalysisP1'    : props.DATA_FINAL_WORK + '/eotbias/metopa/$curDateP1Year/dbias.$curDateP1',
      'satANightAnalysisP1'  : props.DATA_FINAL_WORK + '/eotbias/metopa/$curDateP1Year/nbias.$curDateP1',
      'satBDayAnalysisP1'    : props.DATA_FINAL_WORK + '/eotbias/metopb/$curDateP1Year/dbias.$curDateP1',
      'satBNightAnalysisP1'  : props.DATA_FINAL_WORK + '/eotbias/metopb/$curDateP1Year/nbias.$curDateP1',

      # bias current run day - 1
      'satADayAnalysisM1'    : props.DATA_FINAL_WORK + '/eotbias/metopa/$curDateM1Year/dbias.$curDateM1',
      'satANightAnalysisM1'  : props.DATA_FINAL_WORK + '/eotbias/metopa/$curDateM1Year/nbias.$curDateM1',
      'satBDayAnalysisM1'    : props.DATA_FINAL_WORK + '/eotbias/metopb/$curDateM1Year/dbias.$curDateM1',
      'satBNightAnalysisM1'  : props.DATA_FINAL_WORK + '/eotbias/metopb/$curDateM1Year/nbias.$curDateM1',

      # current run day corrections
      'satADayCorrected'     : props.DATA_FINAL_WORK + '/eotcor/metopa/$curYear/dsobsc.$curDate',
      'satANightCorrected'   : props.DATA_FINAL_WORK + '/eotcor/metopa/$curYear/nsobsc.$curDate',
      'satBDayCorrected'     : props.DATA_FINAL_WORK + '/eotcor/metopb/$curYear/dsobsc.$curDate',
      'satBNightCorrected'   : props.DATA_FINAL_WORK + '/eotcor/metopb/$curYear/nsobsc.$curDate',

      # eot corrected + 1 day
      'satADayCorrectedP1'   : props.DATA_FINAL_WORK + '/eotcor/metopa/$curDateP1Year/dsobsc.$curDateP1',
      'satANightCorrectedP1' : props.DATA_FINAL_WORK + '/eotcor/metopa/$curDateP1Year/nsobsc.$curDateP1',
      'satBDayCorrectedP1'   : props.DATA_FINAL_WORK + '/eotcor/metopb/$curDateP1Year/dsobsc.$curDateP1',
      'satBNightCorrectedP1' : props.DATA_FINAL_WORK + '/eotcor/metopb/$curDateP1Year/nsobsc.$curDateP1',

      # eot cor. - 1 day
      'satADayCorrectedM1'   : props.DATA_FINAL_WORK + '/eotcor/metopa/$curDateM1Year/dsobsc.$curDateM1',
      'satANightCorrectedM1' : props.DATA_FINAL_WORK + '/eotcor/metopa/$curDateM1Year/nsobsc.$curDateM1',
      'satBDayCorrectedM1'   : props.DATA_FINAL_WORK + '/eotcor/metopb/$curDateM1Year/dsobsc.$curDateM1',
      'satBNightCorrectedM1' : props.DATA_FINAL_WORK + '/eotcor/metopb/$curDateM1Year/nsobsc.$curDateM1',

      # scaled NSRs
      'buoyNSRscaled'     : props.DATA + '/common/static/buoy4sm-nsr-stat-v2-scaled',
      'argoNSRscaled'     : props.DATA + '/common/static/buoy4sm-nsr-stat-v2-scaled',
      'shipNSRscaled'     : props.DATA + '/common/static/ship4sm-nsr-stat-v2-scaled',
      'iceNSRscaled'      : props.DATA + '/common/static/cice4sm-nsr-stat-v2-scaled',
      'satDayNSRscaled'   : props.DATA + '/common/static/day-path4sm-nsr-stat-v2-scaled',
      'satNightNSRscaled' : props.DATA + '/common/static/nte-path4sm-nsr-stat-v2-scaled',

      # grid output
      'buoyGridOut'      : props.DATA_FINAL_WORK + '/grid/buoyship/$curYear12/buoy.$curDate12',
      'argoGridOut'      : props.DATA_FINAL_WORK + '/grid/buoyship/$curYear12/argo.$curDate12',
      'shipGridOut'      : props.DATA_FINAL_WORK + '/grid/buoyship/$curYear12/ship.$curDate12',      
      'iceCon720x360'    : props.DATA_FINAL_WORK + '/grid/ice/con/yyyy/icecon720x360.yyyyMMdd',
      'iceCon1440x720'   : props.DATA_FINAL_WORK + '/grid/ice/con/yyyy/icecon.yyyyMMdd',
      'iceCon'           : props.DATA_FINAL_WORK + '/grid/ice/con/yyyy/icecon.yyyyMMdd',
      'iceCon7day'       : props.DATA_FINAL_WORK + '/grid/ice/con/$curYear_ice/icecon7.$curDate_ice',
      'iceConMed'        : props.DATA_FINAL_WORK + '/grid/ice/con-med/$curYear_ice/icecon-med.$curDate_ice',
      'iceSSTGrads'      : props.DATA_FINAL_WORK + '/grid/ice/ice-sst/$curYear_ice/icesst.$curDate_ice',       

#     'iceCon7day'       : props.DATA_FINAL_WORK + '/grid/ice/con/$curYear/icecon7.$curDate',
#     'iceConMed'        : props.DATA_FINAL_WORK + '/grid/ice/con-med/$curYear/icecon-med.$curDate',
#     'iceSSTGrads'      : props.DATA_FINAL_WORK + '/grid/ice/ice-sst/$curYear/fitonesst.$curDate', 

      'satADayGridOut'   : props.DATA_FINAL_WORK + '/grid/metopa/$curYear12/dgrid.sst.$curDate12',
      'satANightGridOut' : props.DATA_FINAL_WORK + '/grid/metopa/$curYear12/ngrid.sst.$curDate12', 
      'satBDayGridOut'   : props.DATA_FINAL_WORK + '/grid/metopb/$curYear12/dgrid.sst.$curDate12',
      'satBNightGridOut' : props.DATA_FINAL_WORK + '/grid/metopb/$curYear12/ngrid.sst.$curDate12',   
            
      # satellite tmp concat file
      'satSSTCatObs' : props.DATA_FINAL_WORK + '/obs/satsst/$curYear12/satobs.$curJulian12',            

      # firstguess/
      'fgOisst'      : props.DATA_FINAL_OUT + '/oiout/$fgYear/sst4-metopab-eot-finv2.$fgDate',

      # oi out
      'oisst'        : props.DATA_FINAL_OUT + '/oiout/$curYear/sst4-metopab-eot-finv2.$curDate',

      # output
      # checked in NetCDF processing F90
      'netCDFrunType' : '\'FINAL\'',

      # this abbreviation is pulled in by the .gs file as a run descriptor
      'runTypeAbrv' : '\'finv2\'',

      # GrADS
      'gradsOutDir'   : props.DATA_FINAL_OUT + '/map/$curYear',
      'gradsPubDir'   : props.DATA_FINAL_OUT + '/map/published',
      'outputRunDate' : props.TMP + '/final_output_run_date.txt',
      'TotalGS'       : props.GASCRP + '/total.gs',
      'AnomGS'        : props.GASCRP + '/anom.gs',
      'dailyCTL'      : props.TMP + '/finv2.ctl',
#     'dailyCTL'      : props.DATA + '/common/static/finv2.ctl',
      'frPathOnly'    : props.DATA_FINAL_OUT + '/NetCDF/GHRSST/$curYear',
      'frPath'        : props.DATA_FINAL_OUT + '/NetCDF/GHRSST/$curYear/FR-$curDate\'120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.xml\'',

      # NetCDF
      'nomadsCDFpathOnly'    : props.DATA_FINAL_OUT + '/NetCDF/$curYear',
      'nomadsCDFpath'        : props.DATA_FINAL_OUT + '/NetCDF/$curYear/oisst-avhrr-v02r01.$curDate\'.nc\'',
      'nomadsIEEEpath'       : props.DATA_FINAL_OUT + '/NetCDF/$curYear/oisst-avhrr-v02r01.$curDate',
      'ghrsstCDFpathOnly'    : props.DATA_FINAL_OUT + '/NetCDF/GHRSST/$curYear',
      'ghrsstCDFJPLpathOnly' : props.DATA_FINAL_OUT + '/NetCDF/GHRSST/GDS2toNASA-JPL',
      'ghrsstCDFpath'        : props.DATA_FINAL_OUT + '/NetCDF/GHRSST/$curYear/$curDate\'120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.1-fv02.1.nc\'',
      'iceConMedNetcdf'      : props.DATA_FINAL_WORK + '/grid/ice/con-med/$curYear/icecon-med.$curDate',
      'climpath_leap'        : props.DATA + '/common/clim/leap/clmoiv2.$curMonth$curDay',
      'climpath_noleap'      : props.DATA + '/common/clim/noleap/clmoiv2.$curMonth$curDay',
    }
  }
}))


#--------------------------------------------
#
# Shared properties of all OI runs
#
#--------------------------------------------

common = json.loads(json.dumps({
  'climCTL' : props.TMP + '/oiv2clm_1440x720.ctl',
# 'climCTL' : props.DATA + '/common/clim/oiv2clm_1440x720.ctl',
  'quarterMaskIn' : props.DATA + '/common/static/quarter-mask-extend',
  'fileStdevIn'   : props.DATA + '/common/static/stdev1d-coads3-fill',

  # EOT for both 
  'filledModes'  : props.DATA + '/common/static/eot6.damp-zero.ev130.ano.dat',
  'twoDegMask'   : props.DATA + '/common/static/lstags.twodeg.dat',
  'twoDegClim'   : props.DATA + '/common/static/clim.71.00.gdat',
  'modeVariance' : props.DATA + '/common/static/var-mode',

  # statics
  'residualStat'      : props.DATA + '/common/static/residual-stat-v2',
  'cor4smStat'        : props.DATA + '/common/static/cor4sm-stat-v2',
  'pathIncrVar'       : props.DATA + '/common/static/path-incr-var-stat-v2',
  'errorCorStat'      : props.DATA + '/common/static/error-cor-stat-v2',
  'quarterMaskExtend' : props.DATA + '/common/static/quarter-mask-extend',
  'clim4'             : props.DATA + '/common/static/oiclm4.mon',

  # buoy, ship, and ice noise signal ratio files
  'buoyNSR'      : props.DATA + '/common/static/buoy4sm-nsr-stat-v2',
  'argoNSR'      : props.DATA + '/common/static/buoy4sm-nsr-stat-v2',
  'shipNSR'      : props.DATA + '/common/static/ship4sm-nsr-stat-v2',
  'iceNSR'       : props.DATA + '/common/static/cice4sm-nsr-stat-v2',

  # satellite NSR
  'satADayNSR'   : props.DATA + '/common/static/day-path4sm-nsr-stat-v2',
  'satANightNSR' : props.DATA + '/common/static/nte-path4sm-nsr-stat-v2',
  'satBDayNSR'   : props.DATA + '/common/static/day-path4sm-nsr-stat-v2',
  'satBNightNSR' : props.DATA + '/common/static/nte-path4sm-nsr-stat-v2',

  # coverages
  'covName'     : props.TMP + '/cover_buoyship.txt',
  'covNameSatA' : props.TMP + '/cover_sat_A.txt',
  'covNameSatB' : props.TMP + '/cover_sat_B.txt',
  'covNameIce'  : props.TMP + '/cover_ice.txt',
}))

#--------------------------------------------
#
# End of properties
#
#--------------------------------------------

