add_subdirectory(test)

# install headers
install(FILES moes.hh MatrixMult.hh inversePowerIteration.hh qrcol.hh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/moes)
