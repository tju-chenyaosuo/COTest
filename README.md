Welcome to the homepage of COTest!

There are three folders and one file in the root folder, including tool, mask, model and bug.txt. We introduce each of them as follows:

* 1.tool, which contains model generation code, mask generation code, and code for testing LLVM and GCC. The code for testing LLVM and GCC uses a model file in "model" folder and a mask file in "mask" folder. More specifically, there are three files and one folder in this folder:

    1. GCC-COTest.py, is code for GCC testing.
    
    2. LLVM-COTest.py, is code for LLVM testing.

    3. generate_model.py, is used to generate model file. 
    
    4. generate_mask.py, is used to generate mask file.
    
    5. tools, which contains implementations of basic functions in COTest, such as file read-write operation and command invocation.
    
* 2.mask, which contains mask files, and it includes seven files. Each of them is mask file in our evaluation study.  For example, GCC-4.3.0_4.3.5-mask.txt is the mask file in GCC-4.3.0=>GCC-4.3.5 and it can be used in GCC-4.3.5's testing.

* 3.model, which contains files about model. There are 5 model files and 5 normalization files. For each model file, there is a corresponding normalization file to get normalization information.

* 4.bug.txt, which lists unknown bugs detected by COTest.

Thanks!

---

Note: The Csmith used in COTest is a modified version from HiCOND[1], and the modified Csmith can be found in [https://github.com/JunjieChen/HiCOND/tree/master/tool/csmith_recorder](url).

[1] Chen J, Wang G, Hao D, et al. History-guided configuration diversification for compiler test-program generation[C]//2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019: 305-316.
