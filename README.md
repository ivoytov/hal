## Docker ##

Docker compose requires a high number of SSH sessions. 

[https://github.com/docker/compose/issues/6463#issuecomment-694768175]

on the REMOTE server, edit `/etc/ssh/sshd_config` and add this line `MaxSessions 100`
then run `sudo service ssh restart`
