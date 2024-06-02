#!/bin/bash

CreateMNF() {
	local CMErr='none'
	
	if [ -s $1 ];then
		local MD5Ar=( $(md5sum --binary $1 |tr -s ' ' |tr -d '\*') )
		if [ ${#MD5Ar[@]} -eq 2 ];then
			local FSZ=$(find ${1%/*}"/" -name "${1##*/}" -printf "%s")
			if [ ${#FSZ} -gt 0 ];then
				echo ${1##*/},${MD5Ar[0]},$FSZ > $1".mnf"
			else
				CMErr="FileSizeError"
			fi
		else
		CMErr="ChkSumError"
		fi
	else
		CMErr="FileNOTFound"
	fi
	
	if [ $CMErr != 'none' ];then
		echo 'CreateMNF Failed: '$CMErr
	fi
}
