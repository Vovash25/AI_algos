#include "state.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>


// Zad1
class MlynekState : public game_state<MlynekState, std::string>
{
public:
    char board[3][8];
    int pieces_to_place;

    MlynekState()
        : game_state(side_to_move::max_player),
          pieces_to_place(18)
    {
        for(int i=0; i<3; ++i)
            for(int j=0; j<8; ++j)
                board[i][j] = '.';
    }

    MlynekState(side_to_move stm, const char in_board[3][8], int p_left, const std::string& move)
        : game_state(stm), pieces_to_place(p_left)
    {
        set_move(move);
        for(int i=0; i<3; ++i)
            for(int j=0; j<8; ++j)
                board[i][j] = in_board[i][j];
    }

    void set_turn(side_to_move s) { stm = s; }

    // Zad2
    bool czy_mlynek(int i, int j) const
    {
        char p = board[i][j];
        if (p == '.') return false;

        if (j % 2 == 0)
        {
            if (board[i][(j + 1) % 8] == p && board[i][(j + 2) % 8] == p) return true;
            if (board[i][(j + 7) % 8] == p && board[i][(j + 6) % 8] == p) return true;
        }
        else
        {
            if (board[i][(j + 7) % 8] == p && board[i][(j + 1) % 8] == p) return true;
            if (board[0][j] == p && board[1][j] == p && board[2][j] == p) return true;
        }
        return false;
    }

    // Zad3
    void rozwiaz_mlynek(std::vector<MlynekState>& result_moves) const
    {
        char opp = (stm == side_to_move::max_player) ? 'C' : 'B';
        std::vector<std::pair<int, int>> candidates;
        std::vector<std::pair<int, int>> all_opp;

        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 8; ++j) {
                if(board[i][j] == opp) {
                    all_opp.push_back({i, j});
                    if (!czy_mlynek(i, j)) candidates.push_back({i, j});
                }
            }
        }

        if (candidates.empty()) candidates = all_opp;

        for(auto& pos : candidates) {
            char next_b[3][8];
            for(int k=0; k<3; ++k) for(int l=0; l<8; ++l) next_b[k][l] = board[k][l];

            next_b[pos.first][pos.second] = '.';
            std::string nm = get_move() + "x" + std::to_string(pos.first) + std::to_string(pos.second);

            result_moves.emplace_back(!stm, next_b, pieces_to_place, nm);
        }
    }

    std::vector<std::pair<int, int>> get_neighbors(int sq, int idx) const {
        std::vector<std::pair<int, int>> n;
        n.push_back({sq, (idx + 1) % 8});
        n.push_back({sq, (idx + 7) % 8});
        if (idx % 2 != 0) {
            if (sq > 0) n.push_back({sq - 1, idx});
            if (sq < 2) n.push_back({sq + 1, idx});
        }
        return n;
    }

    std::vector<MlynekState> successors() const
    {
        std::vector<MlynekState> moves;
        char me = (stm == side_to_move::max_player) ? 'B' : 'C';

        // Zad4
        if (pieces_to_place > 0)
        {
            for(int i=0; i<3; ++i) {
                for(int j=0; j<8; ++j) {
                    if(board[i][j] == '.')
                    {
                        char next_b[3][8];
                        for(int k=0; k<3; ++k) for(int l=0; l<8; ++l) next_b[k][l] = board[k][l];

                        next_b[i][j] = me;
                        MlynekState temp(stm, next_b, pieces_to_place - 1, std::to_string(i) + std::to_string(j));

                        if(temp.czy_mlynek(i, j)) {
                            temp.rozwiaz_mlynek(moves);
                        } else {
                            temp.stm = !stm;
                            moves.push_back(temp);
                        }
                    }
                }
            }
        }
        // Zad5
        else
        {
            std::vector<std::pair<int, int>> my_pawns;
            for(int i=0; i<3; ++i)
                for(int j=0; j<8; ++j)
                    if(board[i][j] == me) my_pawns.push_back({i, j});

            if (my_pawns.size() < 3) return moves;

            bool flying = (my_pawns.size() == 3);

            for(auto& pos : my_pawns)
            {
                int r = pos.first;
                int c = pos.second;
                std::vector<std::pair<int, int>> targets;

                if (flying) {
                    for(int i=0; i<3; ++i)
                        for(int j=0; j<8; ++j)
                            if(board[i][j] == '.') targets.push_back({i, j});
                } else {
                    auto neighbors = get_neighbors(r, c);
                    for(auto& n : neighbors) {
                        if(board[n.first][n.second] == '.') targets.push_back(n);
                    }
                }

                for(auto& t : targets)
                {
                    char next_b[3][8];
                    for(int k=0; k<3; ++k) for(int l=0; l<8; ++l) next_b[k][l] = board[k][l];

                    next_b[r][c] = '.';
                    next_b[t.first][t.second] = me;

                    std::string move_name = std::to_string(r) + std::to_string(c) + "-" +
                                            std::to_string(t.first) + std::to_string(t.second);

                    MlynekState temp(stm, next_b, 0, move_name);

                    if(temp.czy_mlynek(t.first, t.second)) {
                        temp.rozwiaz_mlynek(moves);
                    } else {
                        temp.stm = !stm;
                        moves.push_back(temp);
                    }
                }
            }
        }
        return moves;
    }

    int heuristic() const { return 0; }

    bool operator==(const MlynekState& other) const {
        if (pieces_to_place != other.pieces_to_place || stm != other.stm) return false;
        for(int i=0; i<3; ++i) for(int j=0; j<8; ++j)
            if(board[i][j] != other.board[i][j]) return false;
        return true;
    }
};

template<> struct std::hash<MlynekState> {
    size_t operator()(const MlynekState& s) const {
        size_t h = 0;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 8; ++j) h = h * 31 + s.board[i][j];
        return h;
    }
};

unsigned long long perft(const MlynekState& state, int depth)
{
    if (depth == 0) return 1;
    auto children = state.successors();
    if (depth == 1) return children.size();
    unsigned long long nodes = 0;
    for (const auto& child : children) nodes += perft(child, depth - 1);
    return nodes;
}

void rysuj_plansze(const MlynekState& s)
{
    auto c = [&](int sq, int idx) { return s.board[sq][idx]; };
    std::cout << "\n  (Pieces to place: " << s.pieces_to_place << ")\n";
    std::cout << "7  " << c(0,0) << "--------" << c(0,1) << "--------" << c(0,2) << "\n";
    std::cout << "   |        |        |\n";
    std::cout << "6  |  " << c(1,0) << "-----" << c(1,1) << "-----" << c(1,2) << "  |\n";
    std::cout << "   |  |     |     |  |\n";
    std::cout << "5  |  |  " << c(2,0) << "--" << c(2,1) << "--" << c(2,2) << "  |  |\n";
    std::cout << "   |  |  |     |  |  |\n";
    std::cout << "4  " << c(0,7) << "--" << c(1,7) << "--" << c(2,7) << "     " << c(2,3) << "--" << c(1,3) << "--" << c(0,3) << "\n";
    std::cout << "   |  |  |     |  |  |\n";
    std::cout << "3  |  |  " << c(2,6) << "--" << c(2,5) << "--" << c(2,4) << "  |  |\n";
    std::cout << "   |  |     |     |  |\n";
    std::cout << "2  |  " << c(1,6) << "-----" << c(1,5) << "-----" << c(1,4) << "  |\n";
    std::cout << "   |        |        |\n";
    std::cout << "1  " << c(0,6) << "--------" << c(0,5) << "--------" << c(0,4) << "\n";
    std::cout << "   a  b  c  d  e  f  g\n";
}

// Zad6
template<typename duration_type=std::chrono::seconds>
void play(side_to_move player, unsigned search_depth, duration_type time_limit)
{
    MlynekState state;
    std::string move;
    std::vector<MlynekState> children;
    std::vector<MlynekState>::const_iterator it;

    // Inicjalizacja wyszukiwania
    auto srch = alpha_beta_searcher<MlynekState, int>(
        search_depth,
        &MlynekState::successors,
        &MlynekState::heuristic,
        time_limit
    );

    while (true)
    {
        rysuj_plansze(state);
        children = state.successors();

        if (children.empty()) {
            std::cout << "Koniec gry! Brak mozliwych ruchow.\n";
            if (state.get_stm() == side_to_move::max_player)
                std::cout << "Wygrywa Czarny (MIN)!\n";
            else
                std::cout << "Wygrywa Bialy (MAX)!\n";
            break;
        }

        std::cout << "Dostepne ruchy: ";
        for(size_t i=0; i<children.size(); ++i) {
            std::cout << children[i].get_move() << (i < children.size()-1 ? ", " : "");
        }
        std::cout << "\n";

        if (state.get_stm() == player)
        {
            bool valid_move = false;
            do
            {
                std::cout << "Twoj ruch (np. 00, 00-01, 00x11): ";
                std::cin >> move;

                it = std::find_if(children.begin(), children.end(),
                    [&move](const auto &s){ return s.get_move() == move; });

                if (it != children.end()) {
                    valid_move = true;
                    state = *it;
                } else {
                    std::cout << "Nieprawidlowy ruch! Sprobuj ponownie.\n";
                }
            } while (!valid_move);
        }
        else
        {
            std::cout << "Komputer mysli...\n";
            srch.perform_search(state);
            move = srch.get_best_move().first;
            int score = srch.get_best_move().second;

            std::cout << "Ruch komputera: " << move << " (ocena: " << score << ")\n";

            it = std::find_if(children.begin(), children.end(),
                [&move](const auto &s){ return s.get_move() == move; });

            if (it != children.end()) {
                state = *it;
            } else {
                std::cerr << "BLAD: Komputer wybral niemozliwy ruch!\n";
                break;
            }
        }
    }
}

int main()
{
    std::cout << "=== PRZYKLAD I (Faza I) ===\n";
    MlynekState p1;
    rysuj_plansze(p1);
    std::cout << "Glebokosc 1 | " << perft(p1, 1) << "\n";
    std::cout << "Glebokosc 2 | " << perft(p1, 2) << "\n";
    std::cout << "Glebokosc 3 | " << perft(p1, 3) << "\n";
    std::cout << "Glebokosc 4 | " << perft(p1, 4) << "\n";

    std::cout << "\n=== PRZYKLAD II (Faza I) ===\n";
    MlynekState p2;
    p2.pieces_to_place = 4;
    p2.set_turn(side_to_move::max_player);

    p2.board[0][0]='.'; p2.board[0][1]='.'; p2.board[0][2]='.';
    p2.board[1][0]='.'; p2.board[1][1]='B'; p2.board[1][2]='B';
    p2.board[2][0]='.'; p2.board[2][1]='.'; p2.board[2][2]='.';
    p2.board[0][7]='.'; p2.board[1][7]='C'; p2.board[2][7]='.';
    p2.board[2][3]='.'; p2.board[1][3]='B'; p2.board[0][3]='.';
    p2.board[2][6]='.'; p2.board[2][5]='C'; p2.board[2][4]='.';
    p2.board[1][6]='.'; p2.board[1][5]='C'; p2.board[1][4]='.';
    p2.board[0][6]='C'; p2.board[0][5]='C'; p2.board[0][4]='B';

    rysuj_plansze(p2);
    std::cout << "Glebokosc 1 | " << perft(p2, 1) << "\n";
    std::cout << "Glebokosc 2 | " << perft(p2, 2) << "\n";
    std::cout << "Glebokosc 3 | " << perft(p2, 3) << "\n";
    std::cout << "Glebokosc 4 | " << perft(p2, 4) << "\n";


    std::cout << "\n=== PRZYKLAD III (Faza II) ===\n";
    MlynekState p3;
    p3.pieces_to_place = 0;
    p3.set_turn(side_to_move::max_player);

    p3.board[0][0]='.'; p3.board[0][1]='C'; p3.board[0][2]='.';
    p3.board[1][0]='B'; p3.board[1][1]='B'; p3.board[1][2]='B';
    p3.board[2][0]='.'; p3.board[2][1]='C'; p3.board[2][2]='.';
    p3.board[0][7]='.'; p3.board[1][7]='.'; p3.board[2][7]='B';
    p3.board[2][3]='B'; p3.board[1][3]='.'; p3.board[0][3]='B';
    p3.board[2][6]='.'; p3.board[2][5]='.'; p3.board[2][4]='.';
    p3.board[1][6]='C'; p3.board[1][5]='.'; p3.board[1][4]='.';
    p3.board[0][6]='B'; p3.board[0][5]='C'; p3.board[0][4]='.';

    rysuj_plansze(p3);
    std::cout << "Glebokosc 1 | " << perft(p3, 1) << "\n";
    std::cout << "Glebokosc 2 | " << perft(p3, 2) << "\n";
    std::cout << "Glebokosc 3 | " << perft(p3, 3) << "\n";
    std::cout << "Glebokosc 4 | " << perft(p3, 4) << "\n";

    play(side_to_move::max_player, 4, std::chrono::seconds(5));

    return 0;
}