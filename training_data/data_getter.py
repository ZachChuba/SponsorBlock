from youtube_transcript_api import YouTubeTranscriptApi
import json
import csv
import math
import os
with open("video_ids.txt") as file:
    ids = file.read().splitlines()
for id in ids:
    print(id)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(id)
        transcript = transcript_list.find_transcript(['en'])
        jso_list = transcript.translate('en').fetch()
        #dump to json
        with open(id+'.json', 'w') as outfile:
            json.dump(jso_list, outfile)
        #read csv
        with open('shit.csv', 'r') as readcsv:
            reader = csv.reader(readcsv)
            for row in reader:
                starttime = row[1]
                endtime = row[2]
        sponsorship = ""
        for line in jso_list:
            #if line["start"] is in range of starttime and endtime
            if int(math.trunc(line["start"])) >= int(math.trunc(float(starttime))):
                if int(math.trunc(line["start"]+line["duration"])) <= int(math.trunc(float(endtime))):
                    sponsorship += line["text"]
                    continue
        
        #if the script file exists already
        if os.path.exists("training_files/not_sponsors/"+id+'_script.txt'):
            #read file
            with open("training_files/not_sponsors/"+id+'_script.txt', 'r') as readfile:
                script = readfile.read()
            #remove sponsor from entire script
            new_script = script.replace(sponsorship, "")
            #write to file
            with open("training_files/not_sponsors/"+id+'_script.txt', 'w') as writefile:
                writefile.write(new_script)
            #write sponsor to file
            with open("training_files/sponsors/"+id+'2_sponsor.txt', 'w') as outfile:
                outfile.write(sponsorship)
        else:
            script = ""
            for line in jso_list:
                script += line["text"] + " "
            #remove sponsor from entire script
            new_script = script.replace(sponsorship, "")
            #write sponsor to file
            with open("training_files/sponsors/"+id+'_sponsor.txt', 'w') as outfile:
                outfile.write(sponsorship)
            #write to file
            with open("training_files/not_sponsors/"+id+'_script.txt', 'w') as writefile:
                writefile.write(new_script)
    #catch exception as e
    except Exception as e:
        print(e)
    