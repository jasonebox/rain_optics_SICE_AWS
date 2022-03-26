# files=("r_TOA*.tif" "OAA.tif" "SAA.tif" "SZA.tif" "OZA.tif")
# files=("SCDA_final.tif")
files=("diagnostic_retrieval.tif")

for file in "${files[@]}"; do
    
    sshpass -p 'A1majestic-varmint-starter-guzzler' rsync --ignore-existing -v --relative 8675309@192.168.127.12:/sice-data/rain_event_082021/mosaic/./*/${file} /media/adrien/Elements/rain_optics_SICE_AWS

     # sshpass -p 'A1majestic-varmint-starter-guzzler' rsync --ignore-existing -v --relative 8675309@192.168.127.12:/sice-data/rain_event_082021/mosaic/./*/${file} /home/adrien/EO-IO/rain_optics_SICE_AWS/SICE_retrievals/data


done
