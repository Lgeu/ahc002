// メモ: 未整理の utils は https://atcoder.jp/contests/ahc001/submissions/20928259

#include<iostream>
#include<iomanip>
#include<vector>
#include<map>
#include<set>
#include<algorithm>
#include<numeric>
#include<limits>
#include<bitset>
#include<functional>
#include<type_traits>
#include<queue>
#include<stack>
#include<array>
#include<random>
#include<utility>
#include<cstdlib>
#include<ctime>
#include<string>
#include<sstream>
#include<chrono>
#include<climits>

// ========================== macroes ==========================

#define NDEBUG


#define rep(i,n) for(ll (i)=0; (i)<(n); (i)++)
#define rep1(i,n) for(ll (i)=1; (i)<=(n); (i)++)
#define rep3(i,s,n) for(ll (i)=(s); (i)<(n); (i)++)

#ifndef NDEBUG
#define ASSERT(expr, ...) \
		do { \
			if(!(expr)){ \
				printf("%s(%d): Assertion failed.\n", __FILE__, __LINE__); \
				printf(__VA_ARGS__); \
				abort(); \
			} \
		} while (false)
#else
#define ASSERT(...)
#endif

#define ASSERT_RANGE(value, left, right) \
    ASSERT((left <= value) && (value < right), \
		"`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var) do{ std::cout << #var << '=' << var << endl; } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template<class T, class S> inline bool chmin(T& m, S q) {
	if (m > q) { m = q; return true; }
	else return false;
}

template<class T, class S> inline bool chmax(T& m, const S q) {
	if (m < q) { m = q; return true; }
	else return false;
}

// クリッピング  // clamp (C++17) と等価
template<class T> inline T clipped(const T& v, const T& low, const T& high) {
	return min(max(v, low), high);
}

// 2 次元ベクトル
template<typename T> struct Vec2 {
	/*
	y 軸正は下方向
	x 軸正は右方向
	回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
	*/
	T y, x;
	constexpr inline Vec2() = default;
	constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
	inline Vec2(const Vec2&) = default;  // コピー
	inline Vec2(Vec2&&) = default;  // ムーブ
	inline Vec2& operator=(const Vec2&) = default;  // 代入
	inline Vec2& operator=(Vec2&&) = default;  // ムーブ代入
	template<typename S> constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
	inline Vec2 operator+(const Vec2& rhs) const {
		return Vec2(y + rhs.y, x + rhs.x);
	}
	inline Vec2 operator+(const T& rhs) const {
		return Vec2(y + rhs, x + rhs);
	}
	inline Vec2 operator-(const Vec2& rhs) const {
		return Vec2(y - rhs.y, x - rhs.x);
	}
	template<typename S> inline Vec2 operator*(const S& rhs) const {
		return Vec2(y * rhs, x * rhs);
	}
	inline Vec2 operator*(const Vec2& rhs) const {  // x + yj とみなす
		return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
	}
	template<typename S> inline Vec2 operator/(const S& rhs) const {
		ASSERT(rhs != 0.0, "Zero division!");
		return Vec2(y / rhs, x / rhs);
	}
	inline Vec2 operator/(const Vec2& rhs) const {  // x + yj とみなす
		return (*this) * rhs.inv();
	}
	inline Vec2& operator+=(const Vec2& rhs) {
		y += rhs.y;
		x += rhs.x;
		return *this;
	}
	inline Vec2& operator-=(const Vec2& rhs) {
		y -= rhs.y;
		x -= rhs.x;
		return *this;
	}
	template<typename S> inline Vec2& operator*=(const S& rhs) const {
		y *= rhs;
		x *= rhs;
		return *this;
	}
	inline Vec2& operator*=(const Vec2& rhs) {
		*this = (*this) * rhs;
		return *this;
	}
	inline Vec2& operator/=(const Vec2& rhs) {
		*this = (*this) / rhs;
		return *this;
	}
	inline bool operator!=(const Vec2& rhs) const {
		return x != rhs.x || y != rhs.y;
	}
	inline bool operator==(const Vec2& rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	inline void rotate(const double& rad) {
		*this = rotated(rad);
	}
	inline Vec2<double> rotated(const double& rad) const {
		return (*this) * rotation(rad);
	}
	static inline Vec2<double> rotation(const double& rad) {
		return Vec2(sin(rad), cos(rad));
	}
	static inline Vec2<double> rotation_deg(const double& deg) {
		return rotation(PI * deg / 180.0);
	}
	inline Vec2<double> rounded() const {
		return Vec2<double>(round(y), round(x));
	}
	inline Vec2<double> inv() const {  // x + yj とみなす
		const double norm_sq = l2_norm_square();
		ASSERT(norm_sq != 0.0, "Zero division!");
		return Vec2(-y / norm_sq, x / norm_sq);
	}
	inline double l1_norm() const {
		return std::abs(x) + std::abs(y);
	}
	inline double l2_norm() const {
		return sqrt(x * x + y * y);
	}
	inline double l2_norm_square() const {
		return x * x + y * y;
	}
	inline double abs() const {
		return l2_norm();
	}
	inline double phase() const {  // [-PI, PI) のはず
		return atan2(y, x);
	}
	inline double phase_deg() const {  // [-180, 180) のはず
		return phase() / PI * 180.0;
	}
	bool is_in_area() const {
		return 0 <= y && y < 50 && 0 <= x && x < 50;
	}
};
template<typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
	return rhs * lhs;
}
template<typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
	os << vec.y << ' ' << vec.x;
	return os;
}

// 乱数
struct Random {
	using ull = unsigned long long;
	ull seed;
	inline Random(ull aSeed) : seed(aSeed) {
		ASSERT(seed != 0ull, "Seed should not be 0.");
	}
	const inline ull& next() {
		seed ^= seed << 9;
		seed ^= seed >> 7;
		return seed;
	}
	// (0.0, 1.0)
	inline double random() {
		return (double)next() / (double)ULLONG_MAX;
	}
	// [0, right)
	inline int randint(const int right) {
		return next() % (ull)right;
	}
	// [left, right)
	inline int randint(const int left, const int right) {
		return next() % (ull)(right - left) + left;
	}
};


// キュー
template<class T, int max_size> struct Queue {
	array<T, max_size> data;
	int left, right;
	inline Queue() : data(), left(0), right(0) {}
	inline Queue(initializer_list<T> init) :
		data(init.begin(), init.end()), left(0), right(init.size()) {}

	inline bool empty() const {
		return left == right;
	}
	inline void push(const T& value) {
		data[right] = value;
		right++;
	}
	inline void pop() {
		left++;
	}
	const inline T& front() const {
		return data[left];
	}
	template <class... Args> inline void emplace(const Args&... args) {
		data[right] = T(args...);
		right++;
	}
	inline void clear() {
		left = 0;
		right = 0;
	}
	inline int size() const {
		return right - left;
	}
};


// スタック
template<class T, int max_size> struct Stack {
	array<T, max_size> data;
	int right;

	inline Stack() : data(), right(0) {}
	inline Stack(const int n) : data(), right(0) { resize(n); }
	inline Stack(initializer_list<T> init) :
		data(init.begin(), init.end()), right(init.size()) {}
	inline Stack(const Stack& rhs) : data(), right(rhs.right) {
		for (int i = 0; i < right; i++) {
			data[i] = rhs.data[i];
		}
	}
	Stack& operator=(const Stack& rhs) {
		right = rhs.right;
		for (int i = 0; i < right; i++) {
			data[i] = rhs.data[i];
		}
		return *this;
	}
	Stack& operator=(Stack&&) = default;
	inline bool empty() const {
		return 0 == right;
	}
	inline void push(const T& value) {
		ASSERT_RANGE(right, 0, max_size);
		data[right] = value;
		right++;
	}
	inline void pop() {
		right--;
		ASSERT_RANGE(right, 0, max_size);
	}
	const inline T& top() const {
		return data[right - 1];
	}
	template <class... Args> inline void emplace(const Args&... args) {
		ASSERT_RANGE(right, 0, max_size);
		data[right] = T(args...);
		right++;
	}
	inline void clear() {
		right = 0;
	}
	inline void resize(const int& sz) {
		ASSERT_RANGE(sz, 0, max_size + 1);
		for (; right < sz; right++) {
			data[right] = T();
		}
		right = sz;
	}
	inline void resize(const int& sz, const T& fill_value) {
		ASSERT_RANGE(sz, 0, max_size + 1);
		for (; right < sz; right++) {
			data[right] = fill_value;
		}
		right = sz;
	}
	inline int size() const {
		return right;
	}
	inline T& operator[](const int n) {
		ASSERT_RANGE(n, 0, right);
		return data[n];
	}
	inline const T& operator[](const int n) const {
		ASSERT_RANGE(n, 0, right);
		return data[n];
	}
	inline T* begin() {
		return (T*)data.data();
	}
	inline T* end() {
		return (T*)data.data() + right;
	}
};


// 時間
inline double time() {
	return static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count()) * 1e-9;
}


// 重複除去
template<typename T> inline void deduplicate(vector<T>& vec) {
	sort(vec.begin(), vec.end());
	vec.erase(unique(vec.begin(), vec.end()), vec.end());
}


template<typename T> inline int search_sorted(const vector<T>& vec, const T& a) {
	return lower_bound(vec.begin(), vec.end(), a) - vec.begin();
}

// ------------------------------------------------------ ここまでライブラリ ---------------------------------------------------------


struct Input {
	int sy, sx;
	array<array<int, 50>, 50> T;  // タイル ID
	array<array<int, 50>, 50> P;  // 得点
	void read() {
		cin >> sy >> sx;
		for (int y = 0; y < 50; y++) {
			for (int x = 0; x < 50; x++) {
				cin >> T[y][x];
			}
		}
		for (int y = 0; y < 50; y++) {
			for (int x = 0; x < 50; x++) {
				cin >> P[y][x];
			}
		}
	}
	inline int tile(const Vec2<int>& p) const {
		return T[p.y][p.x];
	}
	inline int point(const Vec2<int>& p) const {
		return P[p.y][p.x];
	}
};


// 探索で、
// 長く生きたほうが良い
// 最終地点が遠いほどよい
// 端っこを使うほどよい.
// 得点が高いほうが良い.

constexpr int MAX_DEPTH = 20;
using Direction = char;
constexpr array<char, 4> Ds{ 'R', 'D', 'L', 'U' };
constexpr array<Vec2<int>, 4> Dyx{ Vec2<int>{0, 1}, {1, 0}, {0, -1}, {-1, 0} };
struct State {
	const Input& input;
	Stack<Direction, 50 * 50> path;  // 使ってない
	Stack<int, 50 * 50> closed;
	Vec2<int> yx_current;
	Stack<Direction, MAX_DEPTH> dfs_path;
	double eval_score;

	Vec2<int> dfs_start;
	double best_eval_score;
	Stack<Direction, MAX_DEPTH> best_dfs_path;  // これちょっと無駄

	State() = default;
	State(const Input& a_input) : input(a_input), path(), dfs_path(), closed(),
		yx_current(a_input.sy, a_input.sx), eval_score(0.0), dfs_start(), best_eval_score(-1e300) {
		for (int i = 0; i < 2500; i++) {
			closed.push(false);
		}
		closed[a_input.tile(yx_current)] = true;
	}
	
	inline double calc_score_cumulative() {
		double res = 0;
		const Vec2<double> v = yx_current;
		constexpr double COEF_CENTER_L1 = -28.516420920425844;  // OPTIMIZE [-100.0, 100.0]
		constexpr double COEF_CENTER_L1_SQ = 19.01870941056843;  // OPTIMIZE [0.0, 200.0]
		constexpr double COEF_POINT = 99.89383738999402;  // OPTIMIZE [0.0, 100.0]
		auto l1 = (v - Vec2<double>(24.5, 24.5)).l1_norm();
		res += l1 * COEF_CENTER_L1;
		res += l1 * l1 * COEF_CENTER_L1_SQ;
		res += (double)input.point(yx_current) * COEF_POINT;
		return res;
	}
	inline double calc_score(
		const int& depth,
		const Vec2<int>& start,
		const Vec2<int>& end
		) {
		if (depth != MAX_DEPTH) {
			return -1e12;
		}
		constexpr double COEF_DIST = 299.954909386729;  // OPTIMIZE [0.0, 10000.0]
		constexpr double COEF_TWIST = 775084.7816850847;  // OPTIMIZE [0.0, 1000000.0]
		double res = 0;
		res += ((Vec2<double>)(start - end)).l2_norm_square() * COEF_DIST;
		Vec2<double> center{ 24.5, 25.5 };
		double twist = ((Vec2<double>)(start) - center).phase() - ((Vec2<double>)(end) - center).phase();
		if (twist < -PI) {
			twist += 2 * PI;
		}
		else if (twist > PI) {
			twist -= 2 * PI;
		}
		res += twist * COEF_TWIST;
		return res;
	}
	inline int move(const Direction& d) {
		// 得点を返す
		yx_current += Dyx[d];
		path.push(d);
		ASSERT(yx_current.is_in_area(), "invalid move!");
		ASSERT(!closed[input.tile(yx_current)], "invalid move!!");
		closed[input.tile(yx_current)] = true;
		return input.point(yx_current);
	}

	Direction dfs() {
		// dfs で次の行動を選択する
		dfs_start = yx_current;
		best_eval_score = -1e300;
		ASSERT(dfs_path.size() == 0, "dfs_path may be wrong?");
		best_dfs_path.clear();
		dfs(1);
		if (best_dfs_path.size() == 0) {
			return -1;
		}
		return best_dfs_path[0];
	}

	void dfs(const int depth) {
		for (Direction d = 0; d < 4; d++) {
			const auto& dyx = Dyx[d];
			yx_current += dyx;
			if (yx_current.is_in_area() && !closed[input.tile(yx_current)]) {
				// >>> do >>>
				closed[input.tile(yx_current)] = true;
				dfs_path.push(d);
				const double old_eval_score = eval_score;
				eval_score += calc_score_cumulative();
				// <<< do <<<

				if (depth == MAX_DEPTH) {
					eval_score += calc_score(depth, dfs_start, yx_current);
				} else {
					dfs(depth + 1);
				}
				if (chmax(best_eval_score, eval_score)) {
					best_dfs_path = dfs_path;
				}

				// >>> undo >>>
				eval_score = old_eval_score;
				dfs_path.pop();
				closed[input.tile(yx_current)] = false;
				// <<< undo <<<
			}
			yx_current -= dyx;
		}
	}

	void print_answer() {
		for (auto&& d : path) {
			cout << Ds[d];
		}
		cout << endl;
	}

};

double T0;

struct Solver {
	const Input& input;
	State state;
	Solver(const Input& a_input) : input(a_input), state(a_input){
	}
	void solve() {
		while (time()-T0 < 1.9) {
			const Direction d = state.dfs();
			if (d == -1) break;
			state.move(d);
		}
		state.print_answer();
	}
};


int main() {
	T0 = time();
	Input input;
	input.read();

	Solver solver(input);
	solver.solve();

}


