#!/usr/bin/bash
cp wardens /usr/bin/
cp wardens.service /usr/lib/systemd/system/
systemctl enable wardens.service 
tar -xvzf lib.tar.gz 
