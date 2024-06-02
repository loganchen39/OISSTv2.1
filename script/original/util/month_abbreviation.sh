#! /bin/bash

getMonthAbbreviation() {
	# 2-digit numerical representation
	month=$1

	# echo the month through a sed script that will assign a month abbreviation
	# to the variable for the 2-digit month
	echo $month|sed -e "
		 s/01/JAN/
		 s/02/FEB/
		 s/03/MAR/
		 s/04/APR/
		 s/05/MAY/
		 s/06/JUN/
		 s/07/JUL/
		 s/08/AUG/
		 s/09/SEP/
		 s/10/OCT/
		 s/11/NOV/
		 s/12/DEC/"
		 
	# convert letters to CAPITAL
	#ccmon=`echo $ccmon|tr [a-z] [A-Z]`
}

