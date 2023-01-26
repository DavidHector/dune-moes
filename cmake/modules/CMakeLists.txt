set(modules "DuneMoesMacros.cmake" FindARPACK.cmake FindARPACKPP.cmake FindSuperLU.cmake)

install(FILES ${modules} DESTINATION ${DUNE_INSTALL_MODULEDIR})
