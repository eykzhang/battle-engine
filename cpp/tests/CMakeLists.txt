add_executable(be_tests
  test_battle_state.cpp
  test_eval.cpp
  test_action.cpp
  test_forward_model.cpp
  test_mcts.cpp
)
target_link_libraries(be_tests PRIVATE be_core Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(be_tests)
