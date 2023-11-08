getquota_zhome.sh
cd ~ && du -h --max-depth=1 .


tar -zcvf montecarlo.tar.gz montecarlo
tar -xvf montecarlo.tar.gz montecarlo
rsync -aP s230025@login2.gbar.dtu.dk:~/resquivel/RL/Training_notebooks/data/carpole/montecarlo.tar.gz .