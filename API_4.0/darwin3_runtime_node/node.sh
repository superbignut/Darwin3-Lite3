#!/bin/bash

##################################################################################################

if [ $# != 1 ];then
	echo "Please use \" -h \" parameter to learn the operation of script."
	exit 1
fi

##################################################################################################

check_start_status(){

	# 循环遍历检查，直到检查进程启动成功为止
	for i in $(seq 1 $3)
	do

		ProcIDTemp=`ps -ef | grep $2 | grep -v grep | awk '{print $2}'`
		if [ "$ProcIDTemp" != "" ];then
			echo -e "\e[42m[$1:	running]\e[0m"
			return
		else
			sleep 1
		fi

	done
	
	echo "$1 start failure."
}

start_fun_node(){
	ProcID=`ps -ef | grep DarwinOS-Node | grep -v grep | awk '{print $2}'`
	if [ "$ProcID" != "" ];then
		echo -e "\e[42m[DarwinOS-Node:	running]\e[0m"
	else
		# 启动cluster
		cd /usr/local/darwinos/node/bin/
		nohup ./DarwinOS-Node >/dev/null 2>&1 &
		cd - >/dev/null 2>&1
		check_start_status DarwinOS-Node DarwinOS-Node 30
	fi
}

start_fun(){
	echo "################################"
	start_fun_node
	echo "################################"
}

##################################################################################################

check_stop_status(){
	
	# 循环遍历检查，直到检查进程停止成功为止
	for i in $(seq 1 $3)
	do

		ProcIDTemp=`ps -ef | grep $2 | grep -v grep | awk '{print $2}'`
		if [ "$ProcIDTemp" != "" ];then
			sleep 1
		else
			echo -e "\e[41m[$1:	stopped]\e[0m"
			return
		fi

	done
	
	echo "$1 stop failure."
}

stop_fun_node(){
	ProcID=`ps -ef | grep DarwinOS-Node | grep -v grep | awk '{print $2}'`
	if [ "$ProcID" != "" ];then
		ps -ef | grep DarwinOS-Node | grep -v grep | awk '{print $2}' | xargs kill >/dev/null 2>&1
		check_stop_status DarwinOS-Node DarwinOS-Node 30
	else
		echo -e "\e[41m[DarwinOS-Node:	stopped]\e[0m"
	fi
}

stop_fun(){
	echo "################################"
	stop_fun_node
	echo "################################"
}

##################################################################################################

status_fun_proc() {
	ProcID=`ps -ef | grep $2 | grep -v grep | awk '{print $2}'`
	if [ "$ProcID" != "" ];then
		echo -e "\e[42m[$1:	running]\e[0m"
	else
		echo -e "\e[41m[$1:	stopped]\e[0m"
	fi
}

status_fun(){
	echo "################################"
	status_fun_proc DarwinOS-Node DarwinOS-Node
	echo "################################"
}

##################################################################################################

if [ "$1" = "start" ];then
	start_fun
elif [ $1 = "stop" ];then
	stop_fun
elif [ $1 = "restart" ];then
	stop_fun
	start_fun
elif [ $1 = "status" ];then
	status_fun
elif [ $1 = "-v" ];then
	cat /usr/local/darwinos/node/version.cfg
elif [ $1 = "-h" ];then
	echo "The legal parameter of this script contains: start, stop, restart, status, -v."
else
	echo "Please use \" -h \" parameter to learn the operation of script."
fi
