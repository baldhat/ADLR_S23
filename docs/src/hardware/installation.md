
## Ubuntu fixed IP
[configuration guide for fixed IP](https://linuxhint.com/change-ip-address-ubuntu/)

``` $ sudo nano /etc/netplan/*.yaml```

```
network:
    renderer: networkd
    ethernets:
        enp1s0:
            dhcp4: no
            gateway4: 192.168.1.1
            addresses: [192.168.1.22/24]
            nameservers:
                addresses: [192.168.1.1]
    version: 2
```


!!! info "DHCP"
    If the UpBoard is operated in a different network, e.g. for configuration, the IP address can be received again from the DHCP server. For this purpose set ```dhcp4: yes```


## Ubuntu serial output access right

The access rights to the serial interface must be set manually. Either once with: \
``` $ sudo chmod 666 /dev/ttyUSB0```\
\
Or permanently by adding the user to the dialout group. (Changes take effect after restart)\
``` $ sudo usermod -a -G dialout $USER```


!!! info "Serial Monitor (screen)"
    If a serial port is to be viewed e.g. via ssh, the tool ``` $ screen``` can be used. [how to monitor the serial port in linux](https://www.pragmaticlinux.com/2021/11/how-to-monitor-the-serial-port-in-linux/).\
    ``` $ screen /dev/ttyUSB0 1000000```\
    ``` $ screen $PORT $BAUDRATE```\
    to exit enter command mode by pressing ```CLTR+a``` Type ```:quit``` and press ENTER.\

!!! info "List serial ports (julia)"
    The serial ports can be displayed with Julia.\
    ```julia> using LibSerialPort;```\
    ```julia> list_ports()```\
   
