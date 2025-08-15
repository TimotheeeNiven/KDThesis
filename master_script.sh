#!/bin/bash

# argument defaults
language="NONE"
install_dir="code"
article_dir="articles"
tokenized_file="tokenized.swc"
aligned_file="aligned.swc"
model=""
ser=""
dic=""
align_ram=4 # GB
runners=4
jobset="."
# configurable but not via switches
model_url="https://www2.informatik.uni-hamburg.de/nats/pub/SWC/acoustic_model_german.tar.xz"
repo_url="https://bitbucket.org/natsuhh/swc.git"
mary_repo_url="https://bitbucket.org/natsuhh/marytts.git"
mary_install_dir="marytts"
# pipeline steps:
step_install=false
step_download=false
step_gen_prep_jobs=false
step_gen_align_jobs=false
step_execute=false
step_status=false

printHelp() {
    echo "Spoken Wikipedia Utility script"
    echo ""
    echo "Decide which parts to run:"
    echo ""
    echo "-I, --install                      Install everything."
    echo "                                   relevant args: -i"
    echo "-D, --download                     Download article data."
    echo "                                   relevant args: -i -a -l"
    echo "-P, --gen-prep-jobs                Generate jobs for article preparation."
    echo "                                   Extracting transcripts and creating the audio.wav files."
    echo "                                   relevant args: -i -a -l -j"
    echo "-A, --gen-align-jobs               Generate jobs for audio alignment."
    echo "                                   relevant args: -i -a -j -m -g -d -A -r"
    echo "-E, --exec-jobs                    Execute jobs which were generated previously."
    echo "                                   relevant args: -i -j -p"
    echo "--status                           display the amount of jobs that are pending, running, failed, finished"
    echo "                                   relevant args: -j"
    echo ""
    echo "Args explained:"
    echo ""
    echo "-i, --install-dir <directory>      Select where to clone the code repository. Default: $install_dir"
    echo "-a, --article-dir <directory>      Select where to download articles to. Default: $article_dir"
    echo "                                   Generated files will also be written in subdirectories of this dir."
    echo "-l, --language 'german'|'english'|'dutch'  Select which data to download and align. Default: $language"
    echo "-j, --jobset <dir>                 The directory containing the *_jobs dirs."
    echo "-m, --model <dir>                  directory of the acoustic model"
    echo "-g, --g2p <.ser file>              path to an .fst.ser file for g2p conversion"
    echo "-d, --dict <.dic file>             path to a g2p dictionary"
    echo "-o, --align-filename <filename>    Filename for the file containing the generated alignments. Default: $aligned_file"
    echo "-r, --ram <int>                    The amount of ram in GB available to each aligning process. Default: $align_ram"
    echo "-p, --processes <int>              The number of processes.  Usually the number of cores is good here. Default: $runners"
    echo "                                   pay attention that enough memory for that many processes is available!"
}

## parsing arguments

isInSet () {
    local e
    for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
    return 1
}

if [[ $# == 0 ]]; then
    printHelp
    exit
fi

while [[ $# > 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            printHelp
            exit 0
            ;;
        -l|--language)
            language="$2"
            if ! isInSet "$language" "english" "german" "dutch"; then
                echo >&2 "Error: Only 'english', 'german' and 'dutch' are supported languages"
                exit 1
            fi
            shift 2
            ;;
        -i|--install-dir)
            install_dir="$2"
            shift 2
            ;;
        -a|--article-dir)
            article_dir="$2"
            shift 2
            ;;
        -o|--align-filename)
            aligned_file="$2"
            shift 2
            ;;
        -m|--model)
            model="$2"
            shift 2
            ;;
        -g|--g2p)
            ser="$2"
            shift 2
            ;;
        -d|--dict)
            dic="$2"
            shift 2
            ;;
        -r|--ram)
            align_ram="$2"
            shift 2
            ;;
        -p|--processes)
            runners="$2"
            shift 2
            ;;
        -j|--jobset)
            jobset="$2"
            shift 2
            ;;
        -I|--install)
            step_install=true
            shift
            ;;
        -D|--download)
            step_download=true
            shift
            ;;
        -P|--gen-prep-jobs)
            step_gen_prep_jobs=true
            shift
            ;;
        -A|--gen-align-jobs)
            step_gen_align_jobs=true
            shift
            ;;
        -E|--exec-jobs)
            step_execute=true
            shift
            ;;
        --status)
            step_status=true
            shift
            ;;
        *)
            shift # otherwise endless loop if switch is not recognized
            ;;
    esac
done

## check for argument consistency

if [[ $step_gen_align_jobs = true ]]; then
   if [[ $model = "" ]]; then
	   echo >&2 "you need to specify a model to create align jobs."
	   exit 1
   fi
   if [[ $ser = "" ]]; then
	   ser=$model/ser
	   echo >&2 "no g2p model given, defaulting to $ser."
   fi
   if [[ $dic = "" ]]; then
	   dic=$model/dic
	   echo >&2 "no dictionary given, defaulting to $dic."
   fi
fi

## checking dependencies

if [ $step_install = true -o $step_gen_prep_jobs = true -o $step_execute = true ]; then
    command -v java >/dev/null 2>&1 || \
        { echo >&2 "java is required but it's not installed.  Aborting."; exit 1; }
fi
if [ $step_gen_prep_jobs = true ]; then
    command -v python3 >/dev/null 2>&1 || \
        { echo >&2 "python3 is required but it's not installed.  Aborting."; exit 1; }
    command -v sox >/dev/null 2>&1 || \
        { echo >&2 "sox is required but it's not installed.  Aborting."; exit 1; }
fi
if [ $step_install = true -o $step_download = true ]; then
    command -v mono >/dev/null 2>&1 || \
        { echo >&2 "mono is required but it's not installed.  Aborting."; exit 1; }
fi
if [ $step_install = true ]; then
    command -v mvn >/dev/null 2>&1 || \
        { echo >&2 "mvn is required but it's not installed.  Aborting."; exit 1; }
fi


gitClone() {
    if [ ! -d "$install_dir" ]; then
        git clone --recursive "$repo_url" "$install_dir"
        if [ $? -ne 0 ]; then
            echo >&2 "Failed to clone repository"
            exit 1
        fi
    fi
}

# use our own marytts system and install it locally.
installOrUpdateMary() {
	if [[ ! -d $mary_install_dir ]]; then
		git clone $mary_repo_url $mary_install_dir
		if [ $? -ne 0 ]; then
            echo >&2 "Failed to clone mary repository"
            exit 1
        fi
		cd $mary_install_dir
	else
		cd $mary_install_dir
		git pull
	fi
	./gradlew publishToMavenLocal
	cd ..
}

# setupWikiDownloader and setupAligner assume that we are
# in the $install_dir.
setupWikiDownloader() {
    if [ ! -f "WikiDownloader/WikiDownloader.exe" ]; then
        echo "Compiling WikiDownloader"
        cd WikiDownloader
        ./make.sh
        if [ ! -f "WikiDownloader.exe" ]; then
            echo >&2 "Failed to build WikiDownloader, exiting."
            exit 1
        fi
        cd ..
    fi
}

setupAligner() {
    if [ ! -f Aligner/target/Aligner.jar ]; then
        echo >&2 "Building Aligner"
        cd Aligner
        mvn clean package
        if [ $? -ne 0 ]; then
            echo >&2 "Failed to build Aligner, exiting."
            exit 1
        fi
        cd ..
    else
        echo >&2 "Aligner built already (Aligner/target/Aligner.jar), skipping ..."
    fi
}

setupModel() {
    if [ ! -d "model_de" ]; then
        echo "Downloading acoustic model, dictionary and G2P model"
        wget $model_url
        tar -xf acoustic_model_german.tar.xz
        rm acoustic_model_german.tar.xz
        mv wiki_v6 model_de
    fi
}

createJobDirs() {
    mkdir -p "$jobset/pending_jobs" "$jobset/running_jobs" "$jobset/finished_jobs" "$jobset/failed_jobs"
}



createPreparationJobs() {
    createJobDirs
    # prepare parameters for later
    if [ "$language" = "german" ]; then
        extraOptions=""
        tokLang="de"
    elif [ "$language" = "english" ]; then
        extraOptions=""
        tokLang="en"
    elif [ "$language" = "dutch" ]; then
        extraOptions="-i -a -n"
        tokLang="de"
	else
		echo >&2 "you need to specify a language to create preparation jobs."
		exit 1
    fi
    ls -1 "$article_dir" | \
        while read dir; do
            if [ -f "$article_dir/$dir/audio.wav" ]; then
                echo "skipping audio processing in job for ${dir}, audio.wav already exists."
                AUDIOJOB=''
            else
                echo "including audio processing in job for ${dir}."
#                cat > "$jobset/pending_jobs/prep_audio_${dir}_job.sh" <<EOF
                AUDIOJOB="python3 $install_dir/prepare_audio.py '"$article_dir/$dir"' 2> /dev/null"
            fi
			tokenize=1
            if [[ -f "$article_dir/$dir/$tokenized_file" ]]; then
                echo "skipping tokenization job for ${dir}, $tokenized_file already exist."
				tokenize=0
			fi
			if [[ $tokenize -eq 0 && -z $AUDIOJOB ]]; then
				continue
			fi
            echo "generating transcript job for ${dir}."
			job_file="$jobset/pending_jobs/prep_${language}_${dir}_job.sh"
            cat > $job_file <<-EOF
				#!/bin/bash
				echo "Job '${dir}' started on '\$(hostname)' at \$(date +"%T")"
				cd "$(pwd)"
				$AUDIOJOB
				EOF
			if [[ $tokenize -eq 1 ]]; then
				cat > $job_file <<-EOF
					java -ea -jar $install_dir/Aligner/target/Aligner.jar tokenize \\
					$extraOptions \\
 					"$article_dir/$dir" \\
					$tokLang \\
					"$article_dir/$dir/$tokenized_file"
					EOF
			fi
        done
}


createJobs() {
	if ! isInSet "$language" "english" "german" "dutch"; then
		echo >&2 "you need to specify a language to create alignment jobs."
		exit 1
	fi
	   
    createJobDirs
    # check if a job should be generated
    ls -1 "$article_dir" | \
        while read dir; do
            if [ -f "$article_dir/$dir/$aligned_file" ]; then
                echo "skipping ${dir}; $aligned_file already exists."
            elif [ ! -f "$article_dir/$dir/audio.wav" ]; then
                echo "skipping ${dir}; audio missing."
            elif [ ! -f "$article_dir/$dir/$tokenized_file" ]; then
                echo "skipping ${dir}; tokenized transcript missing."
            else
                echo "generating job for $dir"
                cat > "$jobset/pending_jobs/align_${language}_${dir}_job.sh" <<EOF
#!/bin/bash
echo "Job '${dir}' started on '\$(hostname)' at \$(date +"%T")"
cd "$(pwd)"

align_ram=$align_ram

java -Xmx\${align_ram}G -Xms\${align_ram}G -jar $install_dir/Aligner/target/Aligner.jar align \\
"$model" \\
"$ser" \\
"$dic" \\
"$article_dir/$dir/audio.wav" \\
"$article_dir/$dir/$tokenized_file" \\
"$article_dir/$dir/$aligned_file"
EOF
            fi
        done
}


executeJobs() {
    echo "Starting aligning process. runners: $runners, jobset: $jobset"
    waitpids=""
    for i in $(seq $runners); do
        sleep $( echo $RANDOM / 10000 | bc -l ) # avoid initial race condition between job runners
        bash $install_dir/jobrunner.sh "$jobset" | sed "s/^/runner${i}: /" &
        waitpids="$! $waitpids"
    done
    wait $waitpids
}


printStatus() {
    echo "pending " $(ls "$jobset/pending_jobs" | wc -l)
    echo "running " $(ls "$jobset/running_jobs" | wc -l)
    echo "failed  " $(ls "$jobset/failed_jobs" | wc -l)
    echo "finished" $(ls "$jobset/finished_jobs" | wc -l)
}

### Executing the desired steps

checkRequiredArg() {
    if [[ -z "$1" ]]; then
        >&2 echo "Argument missing: $1"
        exit 1
    fi
}

if [ $step_install = true ]; then
    echo "Executing Step: Install"
	echo "installing/updating maryTTS..."
	installOrUpdateMary
	echo "fetching aligner code..."
    gitClone
    cd "$install_dir"
    setupWikiDownloader
    setupAligner
    setupModel
    cd ..
    rm -f master_script.sh
    ln -s "$install_dir/master_script.sh" .
fi

if [ $step_download = true ]; then
    echo "Executing Step: Download"
    date=`date --iso-8601=minutes`
    mkdir -p logs/
    $install_dir/WikiDownloader/WikiDownloader.exe $install_dir/WikiDownloader/${language}.json "$article_dir" > >(tee logs/download.$date.stdout) 2> >(tee logs/download.$date.stderr)
fi

if [ $step_gen_prep_jobs = true ]; then
    echo "Executing Step: Generate Preparation jobs"
    checkRequiredArg "$jobset"
    createPreparationJobs
fi

if [ $step_gen_align_jobs = true ]; then
    echo "Executing Step: Generate Align Jobs"
    checkRequiredArg "$jobset"
    createJobs
fi

if [ $step_execute = true ]; then
    echo "Executing Step: Execute Jobs"
    checkRequiredArg "$jobset"
    executeJobs
fi

if [ $step_status = true ]; then
    printStatus
fi