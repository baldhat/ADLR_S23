# Quickstart guide

    
1. [Download Julia](http://julialang.org/downloads/) (v1.8.2) I recommend to check for new versions!
    - For headless machines also download the current version from the website. apt-get may provide an old version.

    ```
    $ sudo apt install wget
    ```

    ```
    $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz
    ```

    ```
    $ tar -xvzf julia-1.8.2-linux-x86_64.tar.gz
    ```


2. Install Julia  [Instructions for Official Binaries](https://julialang.org/downloads/platform/#platform_specific_instructions_for_official_binaries)

    - For Windows follow the installation instructions.
    - For Mac, copy Julia into your application folder.
    - For Linux, extract the folder and copy it to ``` /opt``` 

        ```
        $ sudo cp -r julia-1.8.2 /opt/
        ```

        and create a symbolic link to ```julia``` inside the ```/usr/local/bin``` folder:

        ```
        $ sudo ln -s /opt/julia-1.8.2/bin/julia /usr/local/bin/julia
        ```

        

3. Open Julia

    - For Windows by clicking on the new program icon on your desktop or where you specified it in the installation. Or [add the Julia path to your environment variables](https://www.geeksforgeeks.org/how-to-setup-julia-path-to-environment-variable/)
    - For Mac, by clicking on the new program icon.
    - Type ```julia``` in a new terminal.

4. A Julia terminal should open. To test, enter e.g. ``` sqrt(9)```. 
        
    Press ```ctrl + D``` if you want to close it again.

!!! info "access rights"
    The repository is currently not public. The next steps can only be done after I have given you access rights. For this I need your GitHub name (https://github.com/"name" leads to your account). My email address [finn.sueberkrueb@tum.de](finn.sueberkrueb@tum.de).

5. ```$ git clone https://github.com/Finn-Sueberkrueb/flyonic.git``` or with ```$ git clone git@github.com:Finn-Sueberkrueb/flyonic.git```
6. ```$ cd flyonic```
7.  ```$ julia```
7.  ```julia> ]``` to open the Julia package manager.
8. ```(@v1.8) pkg> activate .``` The environment should be changed from ``` (@v1.7) pkg>  ``` to ```(Flyonic) pkg> ``` 
9. ```(Flyonic) pkg> instantiate``` This will take some time

in case there is a problem with electron [follow...](https://www.techomoro.com/how-to-install-and-set-up-electron-on-ubuntu-19-04-disco-dingo/)

10. Press ```ctrl + c``` to exit package manager
11. Finally, we run the example that learns in the drone environment with ```julia>  include("src/examples/reinforcement_learning.jl")```. With ```include``` the content of the file is inserted into the current REPL (read-eval-print loop "the current terminal session") and thus executed. Keep in mind that all includes are now contained in the current REPL. If you want to start fresh press ``` ctrl + D``` to end the current REPL and start a new one with ```$ Julia```.
12. Now the environment should start in the browser and the drone should start learning. This can also take a little longer the first time, since the entire code must be compiled. It's your turn. There are also Jupiter notebooks in the example folder.


## Optional

To use the Jupiter notebooks in the example folder, iJulia must be installed in addition to Jupiter Notebook. To do this, go to the package manager in Julia again with ```]``` and write ```add iJulia```. Now Jupiter Notebook should be able to run the Julia examples.



Depending on your personal taste, you can setup your favorite IDE. I recommend [Visual Studio Code](https://code.visualstudio.com/docs/languages/julia) with the [Julia extension](https://www.julia-vscode.org). But of course you can also choose from many other options Vim, Jupyter Notebook, Pluto Notebook, Atom, ...