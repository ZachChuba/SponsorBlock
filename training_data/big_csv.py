import csv

#read csv file vipUsers_1636842065895.csv
with open('vipUsers_1636842065895.csv', 'r') as csvvip:
    csv_vip = csv.reader(csvvip)
    with open('sponsorTimes_1636842065895.csv', 'r') as sponsor:
        csv_sponsor = csv.reader(sponsor)
        for sponsor in csv_sponsor:
            for vip in csv_vip:
                print(vip[0], sponsor[7])
                if vip[0] == sponsor[7]:
                    #append sponsor[0] to file
                    with open('alldata.csv', 'a') as alldata:
                        alldatathing = csv.writer(alldata)
                        alldatathing.writerow(sponsor)