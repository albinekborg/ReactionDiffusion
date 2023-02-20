diffusionreaction: diffusionreaction.c
	gcc -O3 -march=native -funroll-loops diffusionreaction.c -o diffusionreaction.exe -IC:/libs/SDL2-2.26.3/x86_64-w64-mingw32/include -LC:/libs/SDL2-2.26.3/x86_64-w64-mingw32/lib -lmingw32 -lSDL2main -lSDL2 -mwindows  -Wl,--no-undefined -Wl,--dynamicbase -Wl,--nxcompat -Wl,--high-entropy-va -lm -ldinput8 -ldxguid -ldxerr8 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lshell32 -lsetupapi -lversion -luuid -static-libgcc -Wall -Wextra
clean:
	rm -f diffusionreaction.exe

