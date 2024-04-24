#!/bin/bash
folders=(
hapod
tests
)

echo "processing folders:"
echo "${folders[@]}"
echo ""

home=$( pwd )
log="code_review.log"

rm -f ${log}

for d in ${folders[@]}
do
    if [[ -d "${d}" ]]
    then
	    echo "code folder ${d}"
	    find ${d} -name "*.py" -exec yapf -i {} + | tee -a ${log}
	    find ${d} -name "*.py" -exec pylint {} + | tee -a ${log}

	    #cd ${d}
	    #pydoctest --include-paths "../pyironth/*.py"
	    #cd ${home}

	    echo ""
	fi
done

#extract most grave notifications
code_notes=(
use-list-literal
unused-variable
unused-import
undefined-all-variable
function-redefined
unused-argument
no-self-argument
undefined-loop-variable
consider-iterating-dictionary
consider-using-dict-items
redefined-builtin
consider-using-with
chained-comparison
reimported
unused-wildcard-import
)

for note in ${code_notes[*]}
do
	grep ${note} ${log}
done
