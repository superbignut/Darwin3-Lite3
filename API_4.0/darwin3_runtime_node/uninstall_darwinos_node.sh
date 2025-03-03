#!/bin/bash

# 卸载node
darwinos_node_uninstall(){
	echo "#### start to uninstall DarwinOS-Node ...####"
	rm -rf /usr/local/darwinos/node
	rm -rf /usr/lib/libzmq*
	echo "#### end to uninstall DarwinOS-Node ...####"
}

# 卸载darwinos node
node_uninstall(){

	# 停止服务
	node.sh stop
	darwinos_node_uninstall

	dir='/usr/local/darwinos'
	if [ -z "$(ls -A $dir)" ]; then
		rm -rf $dir
	fi
	
	rm -rf /usr/local/bin/node.sh
	rm -rf /usr/local/bin/uninstall_darwinos_node.sh
	
	echo "Darwinos node uninstall successfully."
}

if [ ! -d "/usr/local/darwinos/node/" ];then
	echo "Darwinos node is not installed on this machine."
else
	node_uninstall
fi