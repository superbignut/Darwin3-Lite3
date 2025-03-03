#!/bin/bash

# 安装node
darwinos_node_install(){
	echo "#### start to install DarwinOS-Node ...####"
	
	# 拷贝bin文件
	mkdir -p /usr/local/darwinos/node
	mkdir -p /usr/local/darwinos/node/bin
	chmod +x ./bin/DarwinOS-Node
	cp -p ./bin/DarwinOS-Node /usr/local/darwinos/node/bin/
	echo "bin file is ok..."
	
	# 拷贝配置文件
	cp -p ./config/config.ini /usr/local/darwinos/node/bin/
	echo "config file is ok..."
	
	# 拷贝驱动文件
	cp -rp ./script /usr/local/darwinos/node/
	echo "script file is ok..."
	
	# 拷贝lib文件
	cp -p ./lib/libzmq* /usr/lib/
	echo "lib file is ok..."
	
	echo "#### end to install DarwinOS-Node ...####"
}



# 安装darwinos node
node_install(){

	mkdir -p /usr/local/darwinos

	darwinos_node_install
	
	# 将脚本拷贝到服务器
	mkdir -p /usr/local/bin
	cp ./node.sh /usr/local/bin/
	cp ./uninstall_darwinos_node.sh /usr/local/bin/
	
	# 拷贝版本信息
	cp ./version.cfg /usr/local/darwinos/node/
	
	echo "Darwinos node install successfully."
	
	# 启动服务
	node.sh start
}

if [ ! -d "/usr/local/darwinos/node/" ];then
	node_install
else
	echo "Darwinos node is already installed on this machine, please uninstall it firstly."
fi