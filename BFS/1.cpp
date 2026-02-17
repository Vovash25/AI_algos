#include "state.hpp"
#include <iostream>
#include <array>
#include <vector>
#include <unordered_map>
#include <string_view>

class rubik
{
public:
    rubik() : kostka{
        'G', 'G', 'G', 'G',
        'R', 'R', 'R', 'R',
        'B', 'B', 'B', 'B',
        'O', 'O', 'O', 'O',
        'W', 'W', 'W', 'W',
        'Y', 'Y', 'Y', 'Y'
    }
    {}

    bool operator ==(const rubik& s) const { return kostka == s.kostka; }

    size_t hashcode() const {
        return std::hash<std::string_view>{}(std::string_view(kostka.data(), kostka.size()));
    }
    static std::vector<rubik> rotation(const rubik &s)
    {
        static const std::unordered_map<int, int> rl{
            {17, 0}, {16, 2}, {0, 22}, {2, 23},
            {22, 11}, {23, 10}, {11, 17}, {10, 16},
            {13, 12}, {12, 14}, {14, 15}, {15, 13}
        };
        static const std::unordered_map<int, int> rb{
            {12,23},  {23,7},   {7,19},   {19,12},
            {13,21},  {21,5},   {5,17},   {17,13},
            {11,9},   {9,8},    {8,10},   {10,11}
        };
        static const std::unordered_map<int, int> rd{
            {2,6},   {6,9},   {9,13},   {13,2},
            {3,7},   {7,11},  {11,15},   {15,3},
            {22,20}, {20,21}, {21,23},  {23,22}
        };
        std::vector<rubik> res;
        for (const auto &r : {rl, rb, rd})
        {
            auto cw = s, ccw = s;
            for (auto [from, to] : r)
            {
                cw.kostka[to] = s.kostka[from];
                ccw.kostka[from] = s.kostka[to];
            }
            res.push_back(cw);
            res.push_back(ccw);
        }
        return res;
    }
private:
    std::array<char, 24> kostka;
};

template<>
struct std::hash<rubik>
{
    size_t operator ()(const rubik& s) const { return s.hashcode(); }
};

int main()
{
    graph_searcher<rubik, uint8_t, delta_zero, no_heuristic>
        srch(rubik{}, rubik::rotation);
    std::cout << srch.get_summary() << std::endl;
    return 0;
}
