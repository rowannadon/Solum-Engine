#include <array>
#include <cstddef>
#include <utility>


template <std::size_t N, class F, std::size_t... I>
constexpr auto make_array_impl(F&& f, std::index_sequence<I...>) {
  return std::array{ f(std::integral_constant<std::size_t, I>{})... };
}

template <std::size_t N, class F>
constexpr auto make_array(F&& f) {
  return make_array_impl<N>(std::forward<F>(f), std::make_index_sequence<N>{});
}