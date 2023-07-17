*In compliance with the [APACHE-2.0](https://opensource.org/licenses/Apache-2.0) license: I declare that this version of the program contains my modifications, which can be seen through the usual "git" mechanism.*  


2022-11  
Contributor(s):  
Stefan Jansen  
>RELEASE: v2.3 (#146)- moving to PEP517/8
- from versioneer to setuptools_scm
- package_data to pyproject.toml
- tox.ini to pyproject.toml
- flake8 config to .flake8
-removing obsolete setup.cfg
- update all actions
- talib installs from script
- remove TA-Lib constraint and change quick tests to 3.10
- add windows wheels and streamline workflow
- add GHA retry step
- skip two tests that randomly fail on CI
- skip macos Cpy37 arm64  
>add win compiler path  
>np deps by py version  
>add c compiler  
>retry  
>update talib conda to 4.25  
>add c++ compiler  
>tox.ini to pyproject.toml  
>removing ubuntu deps again  
>set prefix in build; move reqs to host  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2022-05  
Contributor(s):  
Eric Lemesre  
>Fixe wrong link (#102)  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2022-04  
Contributor(s):  
MBounouar  
>MAINT: refactoring lazyval + silence a few warnings (#90)* replace distutils.version with packaging.version

* moved the caching lazyval inside zipline

* silence numpy divide errors

* weak_lru_cache small changes

* silence a few pandas futurewarnings

* fix typo

* fix import  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2022-01  
Contributor(s):  
Norman Shi  
>Fix link to the examples directory. (#71)  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-11  
Contributor(s):  
Stefan Jansen  
>update conda build workflows  
>update docs  
>add conda dependency build workflows  
>shorten headings  
>Add conda dependency build workflows (#70)Adds GH actions to build and upload conda packages for TA-Lib and exchange_calendars.  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-10  
Contributor(s):  
MBounouar  
>MAINT: Update development guidelines (#63)* removed unused sequentialpool

* MAINT:Update dev guide (#10)

* fixed links

* fixed a link and deleted a few lines

* fix

* fix

* fix

* Update development-guidelines.rst  
>ENH: Add support for exchange-calendars and pandas > 1.2.5 (#57)* first step
* Switched to exchange_calendars
* fix pandas import  and NaT
* include note in calendar_utils  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-05  
Contributor(s):  
Stefan Jansen  
>fix src layout  
>PACKAGING adopt src layout  
>TESTS adapt to src layout  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-04  
Contributor(s):  
Stefan Jansen  
>readme formatting  
>multiple cleanups  
>editing headlines  
>DOCS edits  
>retry  
>DOCS refs cleanup  
>conda packaging and upload workflows  
>DOCS review  
>ta-lib conda recipe  
>docs revision  
>manifest update - include tests  
>windows wheel talib test  
>workflow update - rebuild cython  
>conda workflow cleanup  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-03  
Contributor(s):  
Stefan Jansen  
>docs update  
>update from master  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 


2021-02  
Contributor(s):  
Stefan Jansen  
>fixed adjustment test tz info issues  
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

