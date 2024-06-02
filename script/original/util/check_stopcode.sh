#! /bin/sh

function checkStopcode {
	currFile=$1
	stopcode=$2

	echo "$currFile: called exit $stopcode"

	if [ $stopcode -eq 0 ]
		then
		echo "GREEN-SUCCESSFUL: {$currFile} completed"
	fi

	if [ $stopcode -eq 10 ]
		then
		echo "YELLOW-WARNING: ERROR IN {$currFile} $stopcode"
	fi

	if [ $stopcode -eq 100 ]
		then
		exit
	fi

	if [ $stopcode -ne 0 -a $stopcode -ne 10 -a $stopcode -ne 100 ]
		then
		echo "RED-STOP: {$currFile} data error"
		exit
	fi
}
