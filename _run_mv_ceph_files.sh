unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

for MONTH in $(seq 10 1 10)
do
    for DAY in $(seq 18 1 22)
    do
        for HOUR in $(seq 0 1 24)
        do
            DIR=$(printf "2022%02d%02d%02d\n" $MONTH $DAY $HOUR)
            echo $DIR
            /usr/bin/aws s3 mv s3://minedojo/videos_downloading/$DIR s3://minedojo/videos/ --recursive
        done
    done
done