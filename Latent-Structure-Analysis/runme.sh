cwd=${PWD}
echo "Current working directory : ${cwd}"

#Ifrac=("0.5")
#Ifrac=("0.99" "0.95" "0.9" "0.85" "0.8" "0.75" "0.7" "0.65" "0.6" "0.55" "0.5" "0.45" "0.4" "0.35" "0.3" "0.25" "0.2" "0.15" "0.1" "0.05")
Ifrac=("0.1" "0.3" "0.5" "0.7" "0.9" "0.99")
#Ifrac=("0.35" "0.3" "0.25" "0.2" "0.15" "0.1" "0.05")

if [[ -f "runme.log" ]]; then
	rm runme.log
fi

touch "${cwd}/runme.log" 

#jmax=7
#jcount=0

No=1
Co=8
JT='100:00:00'

for ifrac in "${Ifrac[@]}"; do
	echo "======================================" >> "$cwd/runme.log"
	echo "Value of ifrac is ${ifrac}" >> "$cwd/runme.log"
	mkdir "${ifrac}"
	cp RUNM-17-08-2026.py "${ifrac}"
	cp CoRE2019-N2M-77K-mod.csv "${ifrac}"
	cd "${ifrac}"

	# Writing Job file
	echo "#!/bin/bash" > job.sh
	echo "#PBS -N SMED-N2-${ifrac}" >> job.sh
	echo "#PBS -l select=${No}:ncpus=${Co},walltime=${JT}" >> job.sh	
	echo "#PBS -j oe" >> job.sh
	echo "#PBS -e err.sm" >> job.sh
	echo "#PBS -o out.sm" >> job.sh
	echo "" >> job.sh
	echo 'cd $PBS_O_WORKDIR' >> job.sh
	echo "" >> job.sh
	chmod 756 job.sh

	sed -i "s/RUNM() running for Ifrac = 0.5/RUNM() running for Ifrac = ${ifrac}/g" RUNM-17-08-2026.py
	sed -i "s/RUNM(0.5, Amp_T, nprop, nmat, hpg, offset, sample, Amp_T_mean, Amp_T_std, Amp_T_stan, 10)/RUNM(${ifrac}, Amp_T, nprop, nmat, hpg, offset, sample, Amp_T_mean, Amp_T_std, Amp_T_stan, 2)/g" RUNM-17-08-2026.py
	sed -i "s/RUNM() ran successfully for Ifrac = 0.5/RUNM() ran successfully for Ifrac = ${ifrac}/g" RUNM-17-08-2026.py
	
	#nohup python-jl RUNM-28-02-2025-CoRE-2019-Ar.py > out_${ifrac}.log 2> warning_${ifrac}.log &
	#echo "time python-jl RUNM-10-08-2026.py > out_${ifrac}.log 2> warning_${ifrac}.log" >> job.sh
        #echo "time python-jl RUNM-10-08-2026.py --input CoRE2019-N2-77K.csv > out_${ifrac}.log 2> warning_${ifrac}.log" >> job.sh
        echo "python-jl RUNM-17-08-2026.py --input CoRE2019-N2M-77K-mod.csv --sample RANDOM > out_${ifrac}.log 2> warning_${ifrac}.log" >> job.sh

	#qsub job.sh
	
	((jcount++))
	echo "Simulation is running for ifrac = ${ifrac}" >> "$cwd/runme.log"
	cd ..

#       if (( jcount >= jmax )); then
#               wait # Wait for current jobs to finish
#               jcount=0
#       fi

done

#wait
echo "" >> "${cwd}/runme.log"
#echo "All jobs are finished" >> "${cwd}/runme.log"
echo "All jobs are submitted successfully" >> "${cwd}/runme.log"
