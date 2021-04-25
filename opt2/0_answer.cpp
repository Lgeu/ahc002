#include <cmath>
#include <array>
#include <vector>
#include <chrono>
#include <climits>
#include <iostream>
#include <algorithm>
#include <initializer_list>
/*
#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#pragma GCC optimize("unroll-loops")
*/
// 007
// ↓ gstep + line （3 変数追加）
// 009
// ↓ 途中保存する
// した

#define NDEBUG


// (009) 0.012331885200000002, 49383405740.0


//constexpr double DUMMY = 1.234;  // OPTIMIZE [-10.0, 0.0]
constexpr double ANNEALING_A = -4.764964219777016;  // OPTIMIZE [-10.0, 0.0]
constexpr double ANNEALING_B = 6.379199589914459;  // OPTIMIZE [0.0, 7.0]
constexpr double ANNEALING_END = 0.0036366400940570933;  // OPTIMIZE [1e-4, 0.005] LOG
constexpr double ANNEALING_START = 0.1194795995308193;  // OPTIMIZE [0.001, 0.5] LOG
constexpr double GRADIENT_CLIPPING = 4.286479241727656e-05;  // OPTIMIZE [1e-7, 2e-4] LOG
constexpr double GRADIENT_DESCENT_LR = 21599021.185847394;  // OPTIMIZE [100000.0, 50000000.0] LOG
constexpr double GRADIENT_DESCENT_MOMENTUM = 0.1786813648093042;  // OPTIMIZE [0.0, 1.0]
constexpr int GRADIENT_DESCENT_STEPS = 23;  // OPTIMIZE [1, 30]
constexpr double GRADIENT_DESCENT_STEPS_STEPPING_PROGRESS_RATE = 0.9725005215053089;  // OPTIMIZE [0.0, 1.0]
constexpr double LINE_LENGTH_LOSS_STEPPING_PROGRESS_RATE = 0.24936016732460342;  // OPTIMIZE [0.0, 1.0]
constexpr double  LINE_LENGTH_LOSS_WEIGHT = 7.433180025970222e-05;  // OPTIMIZE [1e-8, 1e-4] LOG
constexpr int N_RANDOM_FACE_CHOICE = 3;  // OPTIMIZE [1, 10]

std::string hoge="11"; //OPTIMIZE{"11", "unnn" ,"ooo"}

#ifdef ONLINE_JUDGE
#define NDEBUG
#endif

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


	using namespace std;


double T0;//tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
double TIMES;//tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt


template<class T, class S> inline bool chmin(T& m, const S q) {
	if (m > q) { m = q; return true; }
	else return false;
}

template<class T, class S> inline bool chmax(T& m, const S q) {
	if (m < q) { m = q; return true; }
	else return false;
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

// 2 次元ベクトル
template<typename T> struct Vec2 {
	T x, y;
	constexpr Vec2() {}
	constexpr Vec2(const T& arg_x, const T& arg_y) : x(arg_x), y(arg_y) {}
	template<typename S> constexpr Vec2(const Vec2<S>& v) : x((T)v.x), y((T)v.y) {}
	inline Vec2 operator+(const Vec2& rhs) const {
		return Vec2(x + rhs.x, y + rhs.y);
	}
	inline Vec2 operator+(const T& rhs) const {
		return Vec2(x + rhs, y + rhs);
	}
	inline Vec2 operator-(const Vec2& rhs) const {
		return Vec2(x - rhs.x, y - rhs.y);
	}
	template<typename S> inline Vec2 operator*(const S& rhs) const {
		return Vec2(x * rhs, y * rhs);
	}
	inline Vec2& operator+=(const Vec2& rhs) {
		x += rhs.x;
		y += rhs.y;
		return *this;
	}
	inline bool operator!=(const Vec2& rhs) const {
		if (sizeof(x) == sizeof(double)) {
			return neq(x, rhs.x) || neq(y, rhs.y);
		}
		else {
			return x != rhs.x || y != rhs.y;
		}
	}
	inline bool operator==(const Vec2& rhs) const {
		if (sizeof(x) == sizeof(double)) {
			return eq(x, rhs.x) && eq(y, rhs.y);
		}
		else {
			return x == rhs.x && y == rhs.y;
		}
	}
	inline double l2_norm() const {
		return sqrt(x * x + y * y);
	}
	inline double l2_norm_square() const {
		return x * x + y * y;
	}
};
template<typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
	return rhs * lhs;
}
template<typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
	os << vec.y << ' ' << vec.x;
	return os;
}


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

// スタック  // イテレータとか実装したい　誰か help
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


// 勾配射影法 + Momentum + Gradient Clipping  // 軸ごとに最適化するよりいいのかどうか…
template<int max_n, double (*f)(Stack<double, max_n + 3>&, Stack<double, max_n + 3>&)> struct GradientDescent {
	Stack<double, max_n + 3>& x;
	Stack<double, max_n + 3>& x_min;
	Stack<double, max_n + 3>& x_max;
	Stack<double, max_n + 3> gradient;
	Stack<double, max_n + 3> state;
	double lr, momentum;
	double y, max_x_diff;
	double best_y;
	inline GradientDescent(
		Stack<double, max_n + 3>& arg_x, Stack<double, max_n + 3>& arg_x_min, Stack<double, max_n + 3>& arg_x_max,
		const double arg_lr = 0.1, const double arg_momentum = 0.9
	) :
		x(arg_x), x_min(arg_x_min), x_max(arg_x_max),
		gradient(arg_x.size()), state(arg_x.size()), lr(arg_lr), momentum(arg_momentum),
		y(0.0), max_x_diff(0.0), best_y(1e9)
	{
		ASSERT(arg_x.size() == arg_x_min.size(), "lengths of `x` and `x_min` should be same. x.size()=%d, x_min.size=%d.", arg_x.size(), arg_x_min.size());
		ASSERT(arg_x.size() == arg_x_max.size(), "lengths of `x` and `x_max` should be same. x.size()=%d, x_max.size=%d.", arg_x.size(), arg_x_max.size());
	}
	inline GradientDescent(
		Stack<double, max_n + 3>& arg_x, Stack<double, max_n + 3>& arg_x_min, Stack<double, max_n + 3>& arg_x_max, const Stack<double, max_n + 3>& arg_state,
		const double arg_lr = 0.1, const double arg_momentum = 0.9
	) :
		x(arg_x), x_min(arg_x_min), x_max(arg_x_max),
		gradient(arg_x.size()), state(arg_state), lr(arg_lr), momentum(arg_momentum),
		y(0.0), max_x_diff(0.0), best_y(1e9)
	{
		ASSERT(arg_x.size() == arg_x_min.size(), "lengths of `x` and `x_min` should be same. x.size()=%d, x_min.size=%d.", arg_x.size(), arg_x_min.size());
		ASSERT(arg_x.size() == arg_x_max.size(), "lengths of `x` and `x_max` should be same. x.size()=%d, x_max.size=%d.", arg_x.size(), arg_x_max.size());
	}
	inline void calc_gradient() {
		y = f(x, gradient);  // 勾配と目的関数値を更新
		chmin(best_y, y);
	}
	inline void clip(double& val, const double& val_min, const double& val_max) {
		if (val < val_min) val = val_min;
		if (val > val_max) val = val_max;
	}
	inline void step(const int n_free_variables) {  // 勾配計算済みである必要がある  // なんか余計な引数がついてしまった
		max_x_diff = 0.0;
		for (int idx_x = 0; idx_x < n_free_variables; idx_x++) {
			double x_old = x[idx_x];
			x[idx_x] += -gradient[idx_x] * lr + state[idx_x] * momentum;
			clip(x[idx_x], x_min[idx_x], x_max[idx_x]);
			const double d = x[idx_x] - x_old;
			state[idx_x] = d;
			chmax(max_x_diff, abs(d));

		}
	}
	template<bool debug = false> inline void optimize(const int max_iter = 50, int n_free_variables = -1) {
		if (n_free_variables < 0) n_free_variables = x.size();
		for (int iteration = 0; iteration < max_iter; iteration++) {
			calc_gradient();
			step(n_free_variables);
			if (debug) {
				cout << "iteration=" << iteration << " y=" << y << " max_x_diff=" << max_x_diff << endl;
			}
		}
	}
};

template<class T> inline void p_ixor(T*& a, const T* const b, const T* const c) {
	a = (T*)((uintptr_t)a ^ (uintptr_t)b ^ (uintptr_t)c);
}

using PInt = int;
using Point = Vec2<PInt>;
using CInt = int;
struct Vertex;

struct HEdge {
	//double y;  // 位置関係だけを表すならこれはいらない？
	Vertex* l, * r;
	inline HEdge() : //y(0.0),
		l(nullptr), r(nullptr) {}
};

struct VEdge {
	//double x;
	Vertex* u, * d;
	inline VEdge() : //x(0.0),
		u(nullptr), d(nullptr) {}
};

union Edge {
	HEdge as_he;
	VEdge as_ve;
	Edge() : as_he() {}
};

struct Face {
	Point core;
	Vertex* ul, * ur, * dl, * dr;
	HEdge* u, * d;
	VEdge* l, * r;
	inline Face() :
		core(0, 0), ul(nullptr), ur(nullptr), dl(nullptr), dr(nullptr),
		u(nullptr), d(nullptr), l(nullptr), r(nullptr) {}
	/**
	inline double size() const {
		return (double)(d->y - u->y) * (double)(r->x - l->x);
	}
	/**/
};

struct Vertex {
	enum struct Direction : char {
		U, L, D, R, None
	};
	Direction direction;  // ┬ ├ ┴ ┤
	Face* f1, * f2;  // f1 は左上か右下、f2 は右上か左下
	HEdge* he;
	VEdge* ve;
	inline Vertex() : direction(Direction::None), f1(nullptr), f2(nullptr), he(nullptr), ve(nullptr) {}
};

template<int max_n> struct Board {
	/*
	Face   x n
	Edge   x (n + 3)
	Vertex x (2n + 2)
	くらいの実体があるはず…
	ポインタで参照するので vector 使うのは避ける
	*/
	int n;
	Face as_face;
	Stack<Face, max_n> faces;
	Stack<Edge, max_n + 3> edges;
	Stack<Vertex, 2 * max_n + 2> vertices;

	inline Board() = default;
	inline Board(const Board&) = default;  // 超危ない
	inline Board& operator=(const Board& rhs) = default; // 超危ない

	inline void initialize(const int arg_n) {
		n = arg_n;
		faces.resize(n);
		edges.resize(n + 3);
		vertices.resize(2 * n + 2);
		init_as_face();
	}

	bool sanity_check() const {
		for (int i = 0; i < n; i++) {
			const Face& f = faces[i];
			if (f.ul->f1 != &f) { cout << 0 << " " << i << endl; return false; }
			if (f.dr->f1 != &f) { cout << 1 << endl; return false; }
			if (f.ur->f2 != &f) { cout << 2 << endl; return false; }
			if (f.dl->f2 != &f) { cout << 3 << endl; return false; }
			if (f.ul->he != f.ur->he) { cout << 4 << endl; return false; }
			if (f.dl->he != f.dr->he) { cout << 5 << endl; return false; }
			if (f.ul->ve != f.dl->ve) { cout << 6 << endl; return false; }
			if (f.ur->ve != f.dr->ve) { cout << 7 << endl; return false; }
		}
		return true;
	}

	inline void init_as_face() {
		as_face.core = Point(-222, -222);
		as_face.u = &edges[n - 1].as_he;  // 大丈夫かこれ…
		as_face.d = &edges[n].as_he;
		as_face.l = &edges[n + 1].as_ve;
		as_face.r = &edges[n + 2].as_ve;
		as_face.ul = &vertices[2 * n - 2];
		as_face.ur = &vertices[2 * n - 1];
		as_face.dl = &vertices[2 * n];
		as_face.dr = &vertices[2 * n + 1];
		ASSERT(as_face.u == &edges.data[n - 1].as_he, "pointer bugs?");
		//as_face.u->y = 0.0;
		//as_face.d->y = 1e4;
		//as_face.l->x = 0.0;
		//as_face.r->x = 1e4;
		as_face.u->l = as_face.l->u = as_face.ul;
		as_face.u->r = as_face.r->u = as_face.ur;
		as_face.d->l = as_face.l->d = as_face.dl;
		as_face.d->r = as_face.r->d = as_face.dr;
		as_face.ul->he = as_face.ur->he = as_face.u;
		as_face.dl->he = as_face.dr->he = as_face.d;
		as_face.ul->ve = as_face.dl->ve = as_face.l;
		as_face.ur->ve = as_face.dr->ve = as_face.r;
		as_face.ul->direction = as_face.ur->direction = Vertex::Direction::U;  // これは特に意味ないかも
		as_face.dl->direction = as_face.dr->direction = Vertex::Direction::D;
		as_face.ul->f2 = as_face.ur->f1 = as_face.dl->f1 = as_face.dr->f2 = &as_face;  // 通常とは f1 と f2 が逆
	}
};

template<int max_n> struct MiniBoard {
	Stack<Stack<int, max_n>, max_n> data;
	inline MiniBoard() = default;
	inline MiniBoard(const int size_y, const int size_x, const int fill_value = -222) {
		data.resize(size_y);
		for (CInt y = 0; y < size_y; y++) {
			for (CInt x = 0; x < size_x; x++) {
				data[y].push(fill_value);
			}
		}
	}
	inline int size_y() const {
		return data.size();
	}
	inline int size_x() const {
		return data[0].size();
	}
	inline int& get(const int y, const int x) {
		return data[y][x];
	}
	inline Stack<int, max_n>& operator[](const int y) {
		return data[y];
	}
	inline int& operator[](const Point& p) {
		return data[p.y][p.x];
	}
	inline void print() const {
		for (int y = 0; y < data.size(); y++) {
			const int yy = y;
			for (int x = 0; x < data[y].size(); x++) {
				cout << data[yy][x] << " ";
			}
			cout << endl;
		}
	}
};

struct Solution {
	// 各面について (a, b, c, d) を持つ
	vector<array<PInt, 4>> abcds;
	double score;

	// スコアを計算する
	void calc_score() const {

	}

	// 出力する
	void print() const {
		for (const array<PInt, 4>&abcd : abcds) {
			printf("%d %d %d %d\n", abcd[0], abcd[1], abcd[2], abcd[3]);
		}
	}
};


template<typename T> inline void deduplicate(vector<T>& vec) {
	sort(vec.begin(), vec.end());
	vec.erase(unique(vec.begin(), vec.end()), vec.end());
}

template<typename T> inline int search_sorted(const vector<T>& vec, const T& a) {
	return lower_bound(vec.begin(), vec.end(), a) - vec.begin();
}


struct Input {
	int n;
	struct XYR {
		PInt x, y;
		int r;
		CInt x_rank, y_rank;
	};
	vector<XYR> data;
	vector<PInt> deduplicated_xs, deduplicated_ys;

	void read() {
		cin >> n;
		data.resize(n);
		deduplicated_xs.reserve(n);
		deduplicated_ys.reserve(n);
		for (auto& dat : data) {
			cin >> dat.x >> dat.y >> dat.r;
			deduplicated_xs.push_back(dat.x);
			deduplicated_ys.push_back(dat.y);
		}
		deduplicate(deduplicated_xs);
		deduplicate(deduplicated_ys);
		for (auto& dat : data) {
			dat.x_rank = search_sorted(deduplicated_xs, dat.x);
			ASSERT(dat.x == deduplicated_xs[dat.x_rank], "bugs in search_sorted?");
			dat.y_rank = search_sorted(deduplicated_ys, dat.y);
			ASSERT(dat.y == deduplicated_ys[dat.y_rank], "bugs in search_sorted?");
		}
	}
	const XYR& operator[](const int n) const {
		return data[n];
	}
};

// 目的関数と勾配
template<int max_n> struct ObjectiveFunction {
	static const Board<max_n>* board;
	static const Input* input;
	static double line_loss_weight;
	static inline double f(Stack<double, max_n + 3>& x, Stack<double, max_n + 3>& gradient) {

		//cout << line_loss_weight << endl;
		double res = 0.0;
		ASSERT(x.size() == board->edges.size(), "edges edges edges!");
		ASSERT(input->n == board->faces.size(), "faces faces faces!");
		for (int idx_x = 0; idx_x < x.size(); idx_x++) {
			double& g = gradient[idx_x];
			g = 0.0;  // 勾配初期化
		}
		for (int idx_faces = 0; idx_faces < board->faces.size(); idx_faces++) {
			const Face& face = board->faces[idx_faces];
			const int eu = (Edge*)face.u - &board->edges[0];
			const int ed = (Edge*)face.d - &board->edges[0];
			const int el = (Edge*)face.l - &board->edges[0];
			const int er = (Edge*)face.r - &board->edges[0];
			const double s = (double)(x[ed] - x[eu]) * (double)(x[er] - x[el]);
			const double req_inv = 1.0 / (double)(input->operator[](idx_faces).r);
			if (s * req_inv < 1.0) {
				res += (1.0 - s * req_inv) * (1.0 - s * req_inv);
				// 勾配
				double g = 2.0 * (1.0 - s * req_inv) * ((x[er] - x[el]) * req_inv);
				gradient[eu] += g;
				gradient[ed] -= g;
				g = 2.0 * (1.0 - s * req_inv) * ((x[ed] - x[eu]) * req_inv);
				gradient[el] += g;
				gradient[er] -= g;
			}
		}
		for (int idx_edges = 0; idx_edges < board->edges.size() - 4; idx_edges++) {
			const Edge* edge = &board->edges[idx_edges];
			Edge* e1_ = const_cast<Edge*>(&board->edges[idx_edges]);
			Edge* e2_ = const_cast<Edge*>(&board->edges[idx_edges]);
			p_ixor(e1_, (Edge*)edge->as_he.l->he, (Edge*)edge->as_he.l->ve);
			p_ixor(e2_, (Edge*)edge->as_he.r->he, (Edge*)edge->as_he.r->ve);
			const int e1 = e1_ - &board->edges[0];
			const int e2 = e2_ - &board->edges[0];
			res += (x[e2] - x[e1]) * line_loss_weight;
			gradient[e2] += line_loss_weight;
			gradient[e1] -= line_loss_weight;
		}
		double T_TMP = time();//tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
		res /= (double)input->n;
		for (int idx_x = 0; idx_x < x.size(); idx_x++) {
			double& g = gradient[idx_x];
			g /= (double)input->n;

			constexpr double gradient_clipping = GRADIENT_CLIPPING;
			if (g > gradient_clipping) g = gradient_clipping;
			if (g < -gradient_clipping) g = -gradient_clipping;
		}
		TIMES += time() - T_TMP; //tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
		return res;
	}
};
template<int max_n> const Board<max_n>* ObjectiveFunction<max_n>::board;
template<int max_n> const Input* ObjectiveFunction<max_n>::input;
template<int max_n> double ObjectiveFunction<max_n>::line_loss_weight;


template<int max_n> struct FloorplanState {
	Board<max_n> board;
	MiniBoard<max_n> mini_board;  // これ要らなかったかも
	const Input* input;
	Random* rng;
	Stack<double, max_n> face_scores;
	double score;  // 最小化
	Stack<double, max_n + 3> x_val;
	Stack<double, max_n + 3> x_min;
	Stack<double, max_n + 3> x_max;
	Vertex* last_update;
	Stack<double, max_n + 3> last_x_val;
	Stack<double, max_n + 3> last_x_min;
	Stack<double, max_n + 3> last_x_max;
	Stack<double, max_n + 3> momentum_state;


	inline FloorplanState() {}
	inline FloorplanState& operator=(const FloorplanState& rhs) {  // これ全部コピーする必要は無いだろ
		board = rhs.board;
		//mini_board = rhs.board;
		input = rhs.input;
		rng = rhs.rng;
		//face_scores = rhs.face_scores;
		score = rhs.score;
		x_val = rhs.x_val;
		x_min = rhs.x_min;
		x_max = rhs.x_max;
		//last_update = rhs.last_update;
		//last_x_val = rhs.last_x_val;
		//last_x_min = rhs.last_x_min;
		//last_x_max = rhs.last_x_max;
		momentum_state = rhs.momentum_state;
		return *this;
	}

	inline int e_index(const Edge* const& edge) const {
		return (int)(edge - &board.edges[0]);
	}

	// 適当な初期解を作る  // これコンストラクタでやるべきだったかもしれんな…
	void set_initial_state(const Input& arg_input, Random& arg_rng) {

		ASSERT(board.faces.size() == 0, "board should be empty when `set_initial_state` called.");
		const int& n = arg_input.n;
		input = &arg_input;
		rng = &arg_rng;
		face_scores.resize(n);
		// mini_board と board の一部の設定
		board.initialize(n);
		{
			const int n_edges = board.edges.size();
			x_val.resize(n_edges);
			x_min.resize(n_edges);
			x_max.resize(n_edges, 1e4);
			x_val[n_edges - 4] = x_val[n_edges - 2] = 0.0;
			x_val[n_edges - 3] = x_val[n_edges - 1] = 1e4;
			momentum_state.resize(n_edges);
		}
		int idx_edges = 0;
		int idx_vertices = 0;
		mini_board = MiniBoard<max_n>((int)arg_input.deduplicated_ys.size(), (int)arg_input.deduplicated_xs.size(), -222);
		for (int idx_input = 0; idx_input < n; idx_input++) {
			const Input::XYR& inp = arg_input[idx_input];
			const CInt& cy = inp.y_rank;
			const CInt& cx = inp.x_rank;
			mini_board[cy][cx] = ~idx_input;  // ビット反転、いる？
		}
		for (int idx_input = 0; idx_input < n; idx_input++) {
			const Input::XYR& inp = arg_input[idx_input];
			const CInt& cy = inp.y_rank;
			const CInt& cx = inp.x_rank;
			mini_board[cy][cx] = ~idx_input;  // ビット反転、いる？
			Face& face = board.faces[idx_input];
			face.core.y = inp.y;  // これ別にいらないな…
			face.core.x = inp.x;

			CInt x;
			for (x = cx - 1; x >= 0 && mini_board[cy][x] == -222; x--) {
				mini_board[cy][x] = idx_input;
			}
			for (x = cx + 1; x < mini_board.size_x() && mini_board[cy][x] == -222; x++) {
				mini_board[cy][x] = idx_input;
			}

		}

		// board の設定

		auto babs = [](const auto& a) { return a >= 0 ? a : ~a; };

		// 四隅の設定
		{
			// 左上
			Face* face = &board.faces[babs(mini_board[0][0])];
			face->u = board.as_face.u;
			face->ul = board.as_face.ul;
			ASSERT(board.as_face.ul->f1 == nullptr, "bugs in initialization?");
			//ASSERT(board.as_face.ul == &board.vertices[n - 2], "??????? %d %d", board.as_face.ul, &board.vertices[n - 2]);
			board.as_face.ul->f1 = face;

			// 右上
			face = &board.faces[babs(mini_board[0][mini_board.size_x() - 1])];
			face->u = board.as_face.u;
			face->ur = board.as_face.ur;
			ASSERT(board.as_face.ur->f2 == nullptr, "bugs in initialization?");
			board.as_face.ur->f2 = face;

			// 左下
			face = &board.faces[babs(mini_board[mini_board.size_y() - 1][0])];
			face->d = board.as_face.d;
			face->dl = board.as_face.dl;
			ASSERT(board.as_face.dl->f2 == nullptr, "bugs in initialization?");
			board.as_face.dl->f2 = face;

			// 右下
			face = &board.faces[babs(mini_board[mini_board.size_y() - 1][mini_board.size_x() - 1])];
			face->d = board.as_face.d;
			face->dr = board.as_face.dr;
			ASSERT(board.as_face.dr->f1 == nullptr, "bugs in initialization?");
			board.as_face.dr->f1 = face;
		}

		HEdge* he_u;
		HEdge* he_d = board.as_face.u;
		for (CInt y = 0; y < mini_board.size_y(); y++) {
			he_u = he_d;
			if (y < mini_board.size_y() - 1) {  // 最下段以外
				he_d = &board.edges[idx_edges].as_he;
				idx_edges++;

				// 左端の上と下をつなぐ
				{
					Face* face1 = &board.faces[babs(mini_board[y][0])];
					Face* face2 = &board.faces[babs(mini_board[y + 1][0])];

					// メモリ確保
					Vertex* l = &board.vertices[idx_vertices];
					idx_vertices++;

					// 面を設定
					face1->d = face2->u = he_d;  // 2 回設定することになるが気にするな
					face1->l = face2->l = board.as_face.l;  // 2 回設定することになるが気にするな
					face1->dl = face2->ul = l;

					// 辺を設定
					he_d->l = l;

					// 頂点を設定
					l->direction = Vertex::Direction::L;
					l->ve = board.as_face.l;
					l->he = he_d;
					l->f1 = face2;
					l->f2 = face1;
				}

				// 右端の上と下をつなぐ
				{
					Face* face1 = &board.faces[babs(mini_board[y][mini_board.size_x() - 1])];
					Face* face2 = &board.faces[babs(mini_board[y + 1][mini_board.size_x() - 1])];

					// メモリ確保
					Vertex* r = &board.vertices[idx_vertices];
					idx_vertices++;

					// 面を設定
					face1->d = face2->u = he_d;  // 2 回設定することになるが気にするな
					face1->r = face2->r = board.as_face.r;  // 2 回設定することになるが気にするな
					face1->dr = face2->ur = r;

					// 辺を設定
					he_d->r = r;
					//he_d->y = (double)arg_input.deduplicated_ys[y + 1];  // これ別にここで設定しなくてもいいかも  // いや必要かも

					// 頂点を設定
					r->direction = Vertex::Direction::R;
					r->ve = board.as_face.r;
					r->he = he_d;
					r->f1 = face1;
					r->f2 = face2;
				}
			}
			else {  // 最下段
				he_d = board.as_face.d;
			}


			for (CInt x1 = 0; x1 < mini_board.size_x() - 1; x1++) {
				CInt x2 = x1 + 1;
				const int idx_faces_1 = babs(mini_board[y][x1]);
				const int idx_faces_2 = babs(mini_board[y][x2]);
				if (idx_faces_1 != idx_faces_2) {
					//cout << "?? " << idx_faces_1 << " " << idx_faces_2 << endl;
					Face* face1 = &board.faces[idx_faces_1];
					Face* face2 = &board.faces[idx_faces_2];

					VEdge* ve = &board.edges[idx_edges].as_ve;
					idx_edges++;
					Vertex* u = &board.vertices[idx_vertices];
					idx_vertices++;
					Vertex* d = &board.vertices[idx_vertices];
					idx_vertices++;

					// 面を設定
					face1->r = face2->l = ve;
					face1->ur = face2->ul = u;
					face1->dr = face2->dl = d;
					face1->u = face2->u = he_u;  // 2 回設定することになるがまあ真面目に書くのは面倒なので
					face1->d = face2->d = he_d;  // 2 回設定することになるがまあ真面目に書くのは面倒なので

					// 辺を設定
					ve->u = u;
					ve->d = d;
					//ve->x = (double)arg_input.deduplicated_xs[x2];  // これ別にここで設定しなくてもいいかも  // いや必要かも

					// 頂点を設定
					u->direction = Vertex::Direction::U;
					d->direction = Vertex::Direction::D;
					u->ve = d->ve = ve;
					u->he = he_u;
					d->he = he_d;
					u->f2 = d->f1 = face1;
					u->f1 = d->f2 = face2;
				}
			}
		}
		ASSERT(board.sanity_check(), "ia! ia!");
		// ちゃんと初期化できてるか？
		ASSERT(idx_edges == n - 1, "bugs in initialization? idx_edges=%d, n=%d.", idx_edges, n);
		ASSERT(idx_vertices == 2 * n - 2, "bugs in initialization? idx_vertices=%d, n=%d.", idx_vertices, n);
		for (int i = 0; i < board.faces.size(); i++) {
			ASSERT(board.faces[i].u != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].d != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].l != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].r != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].ul != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].ur != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].dl != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.faces[i].dr != nullptr, "initialization error! i=%d.", i);
		}
		for (int i = 0; i < board.edges.size(); i++) {
			ASSERT(board.edges[i].as_he.l != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.edges[i].as_he.r != nullptr, "initialization error! i=%d.", i);
			//ASSERT(board.edges[i].as_he.y != 0.0 || i == n - 1 || i == n + 1, "initialization error! i=%d.", i);
		}
		for (int i = 0; i < board.vertices.size(); i++) {
			ASSERT(board.vertices[i].direction != Vertex::Direction::None, "initialization error! i=%d.", i);
			ASSERT(board.vertices[i].f1 != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.vertices[i].f2 != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.vertices[i].he != nullptr, "initialization error! i=%d.", i);
			ASSERT(board.vertices[i].ve != nullptr, "initialization error! i=%d.", i);
		}
		ASSERT(board.sanity_check(), "ia! ia!");
	}

	// スコアを計算する
	template<bool debug = false> inline void calc_score(double progress_rate = 1.0) {
		int dim = board.edges.size() - 4;
		/*
		x_val.resize(board.edges.size());
		x_min.resize(board.edges.size());
		x_max.resize(board.edges.size());
		x_val[x_val.size() - 4] = x_val[x_val.size() - 2] = 0.0;
		x_val[x_val.size() - 3] = x_val[x_val.size() - 1] = 1e4;
		*/
		fill(x_min.begin(), x_min.end(), 0.0);
		fill(x_max.begin(), x_max.end(), 1e4);

		// 条件設定する

		// TODO: Board 側で座標・制約をもつ必要があるか？ なさそう
		/*
		for (int i = 0; i < dim; i++) {
			const Edge& edge = board.edges[i];
			x_val[i] = edge.as_he.y;
		}*/
		for (int i = 0; i < board.faces.size(); i++) {  // この処理まあまあ無駄な気もする
			const Face& face = board.faces[i];
			const int idx_d = (Edge*)face.d - &board.edges[0];
			const int idx_u = (Edge*)face.u - &board.edges[0];
			const int idx_r = (Edge*)face.r - &board.edges[0];
			const int idx_l = (Edge*)face.l - &board.edges[0];
			const PInt& core_y = input->operator[](i).y;
			const PInt& core_x = input->operator[](i).x;
			chmax(x_min[idx_d], core_y + 1.0);
			chmax(x_val[idx_d], core_y + 1.0);
			chmin(x_max[idx_u], core_y);
			chmin(x_val[idx_u], core_y);
			chmax(x_min[idx_r], core_x + 1.0);
			chmax(x_val[idx_r], core_x + 1.0);
			chmin(x_max[idx_l], core_x);
			chmin(x_val[idx_l], core_x);
		}
		// 最後の 4 次元は壁なので最適化しない
		/*
		x_val.resize(dim);
		x_min.resize(dim);
		x_max.resize(dim);
		*/

		// feasible か確認する
		for (int i = 0; i < dim; i++) {
			ASSERT(x_max[i] - x_min[i] >= 0.0, "constarints infeasivle! i=%d, x_min[i]=%f, x_max[i]=%f.", i, x_min[i], x_max[i]);
		}

		ObjectiveFunction<max_n>::board = &board;
		ObjectiveFunction<max_n>::input = input;
		ObjectiveFunction<max_n>::line_loss_weight = progress_rate < LINE_LENGTH_LOSS_STEPPING_PROGRESS_RATE ? LINE_LENGTH_LOSS_WEIGHT : 0.0;
		GradientDescent<max_n, ObjectiveFunction<max_n>::f> optimizer(x_val, x_min, x_max, momentum_state, GRADIENT_DESCENT_LR, GRADIENT_DESCENT_MOMENTUM);  // momentum を毎回初期化することになるのはどうなんだろう…？ -> 保存するようにした
		int n_steps = progress_rate < GRADIENT_DESCENT_STEPS_STEPPING_PROGRESS_RATE ? GRADIENT_DESCENT_STEPS / 2 + 1 : GRADIENT_DESCENT_STEPS;
		optimizer.optimize(n_steps, dim);  // 変化量が急に跳ねることがあるのが気になる -> gradient clipping で解決
		score = optimizer.best_y;
		momentum_state = optimizer.state;  // !!!!!!!!!!!
	}

	// 厳密な解を構築する
	template<bool debug = false> inline Solution construct_exact_solution() {
		// 再度最適化
		fill(momentum_state.begin(), momentum_state.end(), 0.0);  // momentum 初期化
		GradientDescent<max_n, ObjectiveFunction<max_n>::f> optimizer(x_val, x_min, x_max, 10000.0, 0.97);
		//GradientDescent<ObjectiveFunction<max_n>::f> optimizer(x_val, x_min, x_max, momentum_state, GRADIENT_DESCENT_LR, GRADIENT_DESCENT_MOMENTUM);
		constexpr int n_steps = 200;
		optimizer.template optimize<debug>(n_steps);  // 変化量が急に跳ねることがあるのが気になる -> gradient clipping で解決
		score = optimizer.y;

		Solution solution;
		solution.abcds.reserve(input->n);

		int n_edges = x_val.size();
		/*
		x_val.resize(n_edges + 4);
		x_val[n_edges] = x_val[n_edges + 2] = 0.0;
		x_val[n_edges + 1] = x_val[n_edges + 3] = 1e4;
		*/
		for (int idx_faces = 0; idx_faces < input->n; idx_faces++) {
			const Face& face = board.faces[idx_faces];
			const int l_min = (int)round(x_val[e_index((Edge*)face.l)]);
			const int u_min = (int)round(x_val[e_index((Edge*)face.u)]);
			const int r_max = (int)round(x_val[e_index((Edge*)face.r)]);
			const int d_max = (int)round(x_val[e_index((Edge*)face.d)]);
			auto f = [](const double& req, const int& s) {
				return (1.0 - min(req, (double)s) / max(req, (double)s)) * (1.0 - min(req, (double)s) / max(req, (double)s));
			};
			int w_best = r_max - l_min, h_best = d_max - u_min;
			const int& req = input->operator[](idx_faces).r;
			double best = f((double)req, w_best * h_best);
			int s, w, h;
			double fs;
			if (w_best < h_best) {
				for (w = w_best; w > 0; w--) {
					h = req / w;
					if (h > d_max - u_min) break;
					s = h * w;
					fs = f((double)req, s);
					if (chmin(best, fs)) { w_best = w; h_best = h; }
					h++;
					if (h > d_max - u_min) break;
					s = h * w;
					fs = f((double)req, s);
					if (chmin(best, fs)) { w_best = w; h_best = h; }
				}
			}
			else {
				for (h = h_best; h > 0; h--) {
					w = req / h;
					if (w > r_max - l_min) break;
					s = h * w;
					fs = f((double)req, s);
					if (chmin(best, fs)) { w_best = w; h_best = h; }
					w++;
					if (w > r_max - l_min) break;
					s = h * w;
					fs = f((double)req, s);
					if (chmin(best, fs)) { w_best = w; h_best = h; }
				}
			}
			const int& core_x = input->operator[](idx_faces).x;
			const int& core_y = input->operator[](idx_faces).y;
			const int l = max(l_min, core_x - w_best + 1);
			ASSERT_RANGE(core_x, l, l + w_best);
			const int u = max(u_min, core_y - h_best + 1);
			ASSERT_RANGE(core_y, u, u + h_best);
			solution.abcds.push_back({ l, u, l + w_best, u + h_best });
		}
		/*
		x_val.resize(n_edges);
		*/
		return solution;
	}

	// 状態を遷移する　呼ぶ前に face_scores が埋まっていたほうが良い
	void update() {
		for (int i = 0; i < 300; i++) {
			Vertex* v = update_trial();
			if (v != nullptr) {
				last_update = v;
				last_x_val = x_val;
				last_x_min = x_min;
				last_x_max = x_max;
				goto ok;
			}
		}
		ASSERT(false, "too many faults in update");
		return;
	ok:
		ASSERT(board.sanity_check(), "ia! ia!");
		return;
	}

	double calc_face_score(const int idx_faces) {
		const Face& face = board.faces[idx_faces];
		const int n_edges = board.edges.size();
		/*
		x_val.resize(n_edges);
		x_val[n_edges - 4] = x_val[n_edges - 2] = 0.0;
		x_val[n_edges - 3] = x_val[n_edges - 1] = 1e4;
		*/
		const double& u = x_val[e_index((Edge*)face.u)];
		const double& d = x_val[e_index((Edge*)face.d)];
		const double& l = x_val[e_index((Edge*)face.l)];
		const double& r = x_val[e_index((Edge*)face.r)];
		/*
		x_val.resize(n_edges - 4);
		*/
		ASSERT(u < d&& l < r, "consistency error.");
		const double s = (d - u) * (r - l);
		const double tmp = (1.0 - s / (double)input->operator[](idx_faces).r);
		return tmp * tmp;
	}

	inline Vertex* update_trial(Vertex* const fixed_vertex = nullptr) {
		/*
		ランダムに面を c 個選んで、一番状態が悪いやつを f1 とする
		f1 の頂点からランダムにひとつ点を取り出して v1 とする
		v1 のメンバになっている f1 でない方の面を f2 とし、 f1 と f2 の間の辺を e1 とする（f2 が board.as_face ならおわり）
		e1 のメンバになっている v1 でない方の頂点を v2 とする
		場合分け
		（v2 のメンバになっている 2 つの面が、 f1 と f2 である場合 ... 仕切りで区切られた状態）
			= を || にする
		（それ以外）
			|. を ^_ にする

		これ嘘かも

		この関数の返り値を、この関数の引数にして呼ぶと状態が戻る（はず）
		*/

		Face* f1, * f2;
		Vertex* v1, * v2;
		Edge* e1, * e2;
		if (fixed_vertex == nullptr) {  // v1 を決める
			constexpr int c = N_RANDOM_FACE_CHOICE;
			double worst_score = -1.0;
			int idx_worst_scored_face = 0;
			for (int i = 0; i < c; i++) {
				const int idx_faces = rng->randint(board.n);
				if (chmax(worst_score, /*face_scores[idx_faces]*/ calc_face_score(idx_faces))) {
					idx_worst_scored_face = idx_faces;
				}
			}
			f1 = &board.faces[idx_worst_scored_face];
			const int r1 = rng->randint(4);
			if (r1 == 0) {
				v1 = f1->ul;
				ASSERT(v1->f1 == f1, "consistency error!");
				if (v1->f2 == &board.as_face) return nullptr;
			}
			else if (r1 == 1) {
				v1 = f1->ur;
				ASSERT(v1->f2 == f1, "consistency error!");
				if (v1->f1 == &board.as_face) return nullptr;
			}
			else if (r1 == 2) {
				v1 = f1->dl;
				ASSERT(v1->f2 == f1, "consistency error!");
				if (v1->f1 == &board.as_face) return nullptr;
			}
			else {
				v1 = f1->dr;
				ASSERT(v1->f1 == f1, "consistency error!");
				if (v1->f2 == &board.as_face) return nullptr;
			}
		}
		else {
			v1 = fixed_vertex;
		}
		if (v1->direction == Vertex::Direction::U) {
			f1 = v1->f1;
			f2 = v1->f2;
			ASSERT(f1->l == f2->r, "consistency error!");
			e1 = (Edge*)f1->l;
			if (f1->d == f2->d) {
				goto pattern_I;
			}
			if (f1->core.y < f2->core.y) {
				/*
				-+--+--+-
				 |  |f1|
				 |f2+--+
				 +--+
				*/
				e2 = (Edge*)f1->d;
				if (x_min[e_index(e2)] <= f2->core.y) {
					v2 = f1->dl;  //
					//chmin(e2.max, f2->core.y);  //
					f2->ul->f1 = f1;
					v1->ve = f2->l;  //
					v1->he = (HEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f2 == f1 || v2->f2 == f2, "consistency error!");  //
					ASSERT(v2->f1 != f1 && v2->f1 != f2, "consistency error!");  //
					//v2->f2 ^= f1 ^ f2;  //
					p_ixor(v2->f2, f1, f2);
					swap(e1->as_ve.u, e2->as_he.l);  //
					f1->l = f2->l;  //
					f2->u = (HEdge*)e2;  //
					swap(f1->ul, f2->ul);  //
					swap(f1->dl, f2->ur);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
			else {
				swap(f1, f2);  // 左右反転
				/*
				-+--+--+-
				 |f1|  |
				 +--+f2|
					+--+
				*/
				e2 = (Edge*)f1->d;
				if (x_min[e_index(e2)] <= f2->core.y) {
					v2 = f1->dr;  //
					//chmin(e2.max, f2->core.y);  //
					f2->ur->f2 = f1;
					v1->ve = f2->r;  //
					v1->he = (HEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f1 == f1 || v2->f1 == f2, "consistency error!");  //
					ASSERT(v2->f2 != f1 && v2->f2 != f2, "consistency error!");  //
					//v2->f1 ^= f1 ^ f2;  //
					p_ixor(v2->f1, f1, f2);
					swap(e1->as_ve.u, e2->as_he.r);  //
					f1->r = f2->r;  //
					f2->u = (HEdge*)e2;  //
					swap(f1->ur, f2->ur);  //
					swap(f1->dr, f2->ul);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
		}
		else if (v1->direction == Vertex::Direction::L) {  // U との違いは x 軸と y 軸反転（v<->h, u<->l, d<->r）
			f1 = v1->f1;
			f2 = v1->f2;
			ASSERT(f1->u == f2->d, "consistency error!");
			e1 = (Edge*)f1->u;
			if (f1->r == f2->r) {
				goto pattern_H;
			}
			if (f1->core.x < f2->core.x) {
				/*
				|
				+----+
				|  f2|
				+--+-+
				|f1|
				+--+
				|
				*/
				e2 = (Edge*)f1->r;
				if (x_min[e_index(e2)] <= f2->core.x) {
					v2 = f1->ur;  //
					//chmin(e2.max, f2->core.x);  //
					f2->ul->f1 = f1;
					v1->he = f2->u;  //
					v1->ve = (VEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f2 == f1 || v2->f2 == f2, "consistency error!");  //
					ASSERT(v2->f1 != f1 && v2->f1 != f2, "consistency error!");  //
					//v2->f2 ^= f1 ^ f2;  //
					p_ixor(v2->f2, f1, f2);
					swap(e1->as_he.l, e2->as_ve.u);  //
					f1->u = f2->u;  //
					f2->l = (VEdge*)e2;  //
					swap(f1->ul, f2->ul);  //
					swap(f1->ur, f2->dl);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
			else {
				swap(f1, f2);  // 上下反転
				/*
				|
				+--+
				|f1|
				+--+-+
				|  f2|
				+----+
				|
				*/
				e2 = (Edge*)f1->r;
				if (x_min[e_index(e2)] <= f2->core.x) {
					v2 = f1->dr;  //
					//chmin(e2.max, f2->core.x);  //
					f2->dl->f2 = f1;
					v1->he = f2->d;  //
					v1->ve = (VEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f1 == f1 || v2->f1 == f2, "consistency error!");  //
					ASSERT(v2->f2 != f1 && v2->f2 != f2, "consistency error!");  //
					//v2->f1 ^= f1 ^ f2;  //
					p_ixor(v2->f1, f1, f2);
					swap(e1->as_he.l, e2->as_ve.d);  //
					f1->d = f2->d;  //
					f2->l = (VEdge*)e2;  //
					swap(f1->dl, f2->dl);  //
					swap(f1->dr, f2->ul);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
		}
		else if (v1->direction == Vertex::Direction::D) {  // U と点対称（u<->d, l<->r, min<->max）
			f1 = v1->f1;
			f2 = v1->f2;
			ASSERT(f1->r == f2->l, "consistency error!");
			e1 = (Edge*)f1->r;
			if (f1->u == f2->u) {
				swap(f1, f2);
				goto pattern_I;
			}
			if (f1->core.y > f2->core.y) {
				/*
					+--+
				 +--+f2|
				 |f1|  |
				-+--+--+-
				*/
				e2 = (Edge*)f1->u;
				if (x_max[e_index(e2)] >= f2->core.y + 1) {
					v2 = f1->ur;  //
					//chmax(e2.min, f2->core.y + 1);  //
					f2->dr->f1 = f1;
					v1->ve = f2->r;  //
					v1->he = (HEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f2 == f1 || v2->f2 == f2, "consistency error!");  //
					ASSERT(v2->f1 != f1 && v2->f1 != f2, "consistency error!");  //
					//v2->f2 ^= f1 ^ f2;  //
					p_ixor(v2->f2, f1, f2);
					swap(e1->as_ve.d, e2->as_he.r);  //
					f1->r = f2->r;  //
					f2->d = (HEdge*)e2;  //
					swap(f1->dr, f2->dr);  //
					swap(f1->ur, f2->dl);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
			else {
				swap(f1, f2);  // 左右反転
				/*
				 +--+
				 |f2+--+
				 |  |f1|
				-+--+--+-
				*/
				e2 = (Edge*)f1->u;
				if (x_max[e_index(e2)] >= f2->core.y + 1) {
					v2 = f1->ul;  //
					//chmax(e2.min, f2->core.y + 1);  //
					f2->dl->f2 = f1;
					v1->ve = f2->l;  //
					v1->he = (HEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f1 == f1 || v2->f1 == f2, "consistency error!");  //
					ASSERT(v2->f2 != f1 && v2->f2 != f2, "consistency error!");  //
					//v2->f1 ^= f1 ^ f2;  //
					p_ixor(v2->f1, f1, f2);
					swap(e1->as_ve.d, e2->as_he.l);  //
					f1->l = f2->l;  //
					f2->d = (HEdge*)e2;  //
					swap(f1->dl, f2->dl);  //
					swap(f1->ul, f2->dr);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
		}
		else if (v1->direction == Vertex::Direction::R) {  // D と点対称（u<->d, l<->r, min<->max）
			f1 = v1->f1;
			f2 = v1->f2;
			ASSERT(f1->d == f2->u, "consistency error!");
			e1 = (Edge*)f1->d;
			if (f1->l == f2->l) {
				swap(f1, f2);
				goto pattern_H;
			}
			if (f1->core.x > f2->core.x) {
				/*
					 |
				  +--+
				  |f1|
				+-+--+
				|f2  |
				+----+
					 |
				*/
				e2 = (Edge*)f1->l;
				ASSERT(e1 != e2, "consistency error!");
				if (x_max[e_index(e2)] >= f2->core.x + 1) {
					v2 = f1->dl;  //
					//chmax(e2.min, f2->core.x + 1);  //
					f2->dr->f1 = f1;
					v1->he = f2->d;  //
					v1->ve = (VEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f2 == f1 || v2->f2 == f2, "consistency error!");  //
					ASSERT(v2->f1 != f1 && v2->f1 != f2, "consistency error!");  //
					//v2->f2 ^= f1 ^ f2;  //
					p_ixor(v2->f2, f1, f2);
					swap(e1->as_he.r, e2->as_ve.d);  //
					f1->d = f2->d;  //
					f2->r = (VEdge*)e2;  //
					swap(f1->dr, f2->dr);  //
					swap(f1->dl, f2->ur);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
			else {
				swap(f1, f2);  // 上下反転
				/*
					 |
				+----+
				|f2  |
				+-+--+
				  |f1|
				  +--+
					 |
				*/
				e2 = (Edge*)f1->l;
				ASSERT(e1 != e2, "consistency error!");
				if (x_max[e_index(e2)] >= f2->core.x + 1) {
					v2 = f1->ul;  //
					//chmax(e2.min, f2->core.x + 1);  //
					f2->ur->f2 = f1;
					v1->he = f2->u;  //
					v1->ve = (VEdge*)e2;  //
					swap(v1->f1, v1->f2);
					swap(v1->direction, v2->direction);
					ASSERT(v2->f1 == f1 || v2->f1 == f2, "consistency error!");  //
					ASSERT(v2->f2 != f1 && v2->f2 != f2, "consistency error!");  //
					//v2->f1 ^= f1 ^ f2;  //
					p_ixor(v2->f1, f1, f2);
					swap(e1->as_he.r, e2->as_ve.u);  //
					f1->u = f2->u;  //
					f2->r = (VEdge*)e2;  //
					swap(f1->ur, f2->ur);  //
					swap(f1->ul, f2->dr);  //
					ASSERT(board.sanity_check(), "ia! ia!");
					return v1;
				}
				else {
					return nullptr;
				}
			}
		}
		else {
			ASSERT(false, "invalid direction.");
		}
		return nullptr;
		{
		pattern_I:
			v1 = e1->as_ve.u;
			v2 = e1->as_ve.d;
			if (f1->core.y < f2->core.y) {
				/*
				+-----+
				|  |f1|
				|f2|  |
				+-----+
				*/
				f1->dr->f1 = f2;
				f2->ul->f1 = f1;
				f1->l = f2->l;
				f2->r = f1->r;
				f1->ul = f2->ul;
				f2->dr = f1->dr;
				f1->d = f2->u = (HEdge*)e1;
				e1->as_he.r = f1->dr = f2->ur = v1;  // 右回転: v->f の変更を回避
				e1->as_he.l = f1->dl = f2->ul = v2;
				v1->direction = Vertex::Direction::R;
				v2->direction = Vertex::Direction::L;
				v2->ve = f2->l;
				v1->ve = f1->r;
				v1->he = v2->he = (HEdge*)e1;
				ASSERT(board.sanity_check(), "ia! ia!");
				return v1;
			}
			else if (f1->core.y > f2->core.y) {  // u<->d
			 /*
			 +-----+
			 |f2|  |
			 |  |f1|
			 +-----+
			 */
				f1->ur->f2 = f2;
				f2->dl->f2 = f1;
				f1->l = f2->l;
				f2->r = f1->r;
				f1->dl = f2->dl;
				f2->ur = f1->ur;
				f1->u = f2->d = (HEdge*)e1;
				e1->as_he.r = f1->ur = f2->dr = v2;  // 左回転
				e1->as_he.l = f1->ul = f2->dl = v1;
				v2->direction = Vertex::Direction::R;
				v1->direction = Vertex::Direction::L;
				v1->ve = f2->l;
				v2->ve = f1->r;
				v1->he = v2->he = (HEdge*)e1;
				ASSERT(board.sanity_check(), "ia! ia!");
				return v1;
			}
			else {
				return nullptr;
			}
		}
		{
		pattern_H:
			// y<->x, l<->u, d<->r
			v1 = e1->as_he.l;
			v2 = e1->as_he.r;
			if (f1->core.x < f2->core.x) {
				/*
				+-----+
				|   f2|
				+-----+
				|f1   |
				+-----+
				*/
				f1->dr->f1 = f2;
				f2->ul->f1 = f1;
				f1->u = f2->u;
				f2->d = f1->d;
				f1->ul = f2->ul;
				f2->dr = f1->dr;
				f1->r = f2->l = (VEdge*)e1;
				e1->as_ve.d = f1->dr = f2->dl = v1;  // 左回転
				e1->as_ve.u = f1->ur = f2->ul = v2;
				v1->direction = Vertex::Direction::D;
				v2->direction = Vertex::Direction::U;
				v2->he = f2->u;
				v1->he = f1->d;
				v1->ve = v2->ve = (VEdge*)e1;
				ASSERT(board.sanity_check(), "ia! ia!");
				return v1;
			}
			else if (f1->core.x > f2->core.x) {  // l<->r
			 /*
			 +-----+
			 |f2   |
			 +-----+
			 |   f1|
			 +-----+
			 */
				f1->dl->f2 = f2;
				f2->ur->f2 = f1;
				f1->u = f2->u;
				f2->d = f1->d;
				f1->ur = f2->ur;
				f2->dl = f1->dl;
				f1->l = f2->r = (VEdge*)e1;
				e1->as_ve.d = f1->dl = f2->dr = v2;  // 右回転
				e1->as_ve.u = f1->ul = f2->ur = v1;
				v2->direction = Vertex::Direction::D;
				v1->direction = Vertex::Direction::U;
				v1->he = f2->u;
				v2->he = f1->d;
				v1->ve = v2->ve = (VEdge*)e1;
				ASSERT(board.sanity_check(), "ia! ia!");
				return v1;
			}
			else {
				return nullptr;
			}
		}

		ASSERT(false, "unreachable!");
		return nullptr;
	}

	// 状態を戻す
	inline void undo() {
		ASSERT(last_update != nullptr, "undo was called before update.");
		x_val = last_x_val;
		x_min = last_x_min;
		x_max = last_x_max;
		Vertex* r = update_trial(last_update);
		ASSERT(r != nullptr, "undo error!!!");
		last_update = nullptr;
	}
};


template<class State> struct SimulatedAnnealing {
	State* state;
	Random* rng;
	double best_score;
	State best_state;

	inline SimulatedAnnealing(State& arg_state, Random& arg_rng) :
		state(&arg_state), rng(&arg_rng), best_score(1e9) {}

	template<double (*temperature_schedule)(const double&)> void optimize(const double time_limit) {
		T0 = time(); TIMES = 0.0;
		const double t0 = time();
		double old_score = state->score;
		int iteration = 0;
		while (true) {
			iteration++;
			const double t = time() - t0;
			if (t > time_limit) break;
			const double progress_rate = t / time_limit;

			state->update();
			state->calc_score(progress_rate);
			const double new_score = state->score;
			if (chmin(best_score, new_score)) {
				//cout << "upd! new_score=" << new_score << " progress=" << progress_rate << endl;
				best_state = *state;  // 中にポインタあるので超危ない（後で戻すのでセーフ）
			}
			const double gain = old_score - new_score;  // 最小化: 良くなったらプラス
			const double temperature = temperature_schedule(t);
			const double acceptance_proba = exp(gain / temperature);
			if (acceptance_proba > rng->random()) {
				// 遷移する
				old_score = new_score;
			}
			else {
				// 遷移しない（戻す）
				state->undo();
			}
		}
		*state = best_state;  // 中にポインタあるので超危ない
		//cout << "update_time=" << TIMES << endl;//tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
		//cout << "iteration=" << iteration << endl;//iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
	}
};

inline double sigmoid(const double& a, const double& x) {
	return 1.0 / (1.0 + exp(-a * x));
}

// f: [0, 1] -> [0, 1]
inline double monotonically_increasing_function(const double& a, const double& b, const double& x) {
	ASSERT(b >= 0.0, "parameter `b` should be positive.");
	// a は -10 ～ 10 くらいまで、 b は 0 ～ 1 くらいまで探せば良さそう

	if (a == 0) return x;
	const double x_left = a > 0 ? -b - 0.5 : b - 0.5;
	const double x_right = x_left + 1.0;
	const double left = sigmoid(a, x_left);
	const double right = sigmoid(a, x_right);
	const double y = sigmoid(a, x + x_left);
	return (y - left) / (right - left);  // left とかが大きい値になると誤差がヤバイ　最悪 0 除算になる  // b が正なら大丈夫っぽい
}

// f: [0, 1] -> [start, end]
inline double monotonic_function(const double& start, const double& end, const double& a, const double& b, const double& x) {
	return monotonically_increasing_function(a, b, x) * (end - start) + start;
}


inline double annealing_temperature_schedule(const double& x) {
	constexpr double start = ANNEALING_START;
	constexpr double end = ANNEALING_END;

	constexpr double a = ANNEALING_A;
	constexpr double b = ANNEALING_B;
	//return monotonic_function(start, end, a, b, x < 1.0/3.0 ? x * 3.0 : x < 2.0/3.0 ? x * 3.0 - 1.0 : x * 3.0 - 2.0);
	//return monotonic_function(start, end, a, b, x < 0.5 ? x * 2.0 : x * 2.0 - 1.0);
	return monotonic_function(start, end, a, b, x);
}


struct Tester {
	void test_initialization() {
		Input input;
		input.read();
		Random rng(42);

		FloorplanState<200> floorplan_state;
		floorplan_state.set_initial_state(input, rng);

		cout << "OK!" << endl;
	}

	static double simple_quad(vector<double>& x, vector<double>& gradient) {
		gradient[0] = 2 * x[0];
		return x[0] * x[0];
	}
	static double rosenbrock(vector<double>& x, vector<double>& gradient) {
		double a = 100.0;
		double y = a * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
		gradient[0] = 4 * a * x[0] * x[0] * x[0] - 4 * a * x[0] * x[1] + 2 * x[0] - 2;
		gradient[1] = 2 * a * (x[1] - x[0] * x[0]);
		return y;
	}
	/*
	void test_gradient_descent() {
		/**
		vector<double> x{ 5.0, 5.0 };
		vector<double> x_min{ -5.0, -5.0 };
		vector<double> x_max{ 5.0, 5.0 };
		GradientDescent<rosenbrock> gd(x, x_min, x_max, 0.00001, 0.5);
		/**
		vector<double> x{ 5.0 };
		vector<double> x_min{ -5.0 };
		vector<double> x_max{ 5.0 };
		GradientDescent<simple_quad> gd(x, x_min, x_max, 0.1, 0.9);
		/**
		gd.optimize<true>();
		cout << "OK!" << endl;
	}*/

	void test_initial_state_gradient_descent() {
		Input input;
		input.read();

		Random rng(42);

		FloorplanState<200> floorplan_state;
		floorplan_state.set_initial_state(input, rng);
		floorplan_state.calc_score<true>();

		cout << "OK!" << endl;
	}

	void test_some_steps() {
		Input input;
		input.read();

		Random rng(42);

		FloorplanState<200> floorplan_state;
		cout << "initializing..." << endl;
		floorplan_state.set_initial_state(input, rng);
		cout << "ok." << endl;
		cout << "score calculating..." << endl;
		floorplan_state.calc_score<true>();
		cout << "ok." << endl;

		for (int i = 1; i <= 100; i++) {
			cout << "[step " << i << "] state updating..." << endl;
			floorplan_state.update();
			cout << "[step " << i << "] ok." << endl;
			cout << "[step " << i << "] score calculating..." << endl;
			floorplan_state.calc_score<true>();
			cout << "[step " << i << "] ok." << endl;
			floorplan_state.undo();
		}
		cout << "OK!" << endl;
	}

	void test_monotonic_function() {
		while (1) {
			double a = 10.0, b = 0.0;
			cout << "a = ";
			cin >> a;
			cout << "b = ";
			cin >> b;
			const double start = 100, end = 1;
			const int x_resolution = 40;
			for (int i = 0; i <= x_resolution; i++) {
				const double x = (double)i / (double)x_resolution;
				const double y = monotonic_function(start, end, a, b, x);
				for (int j = 0; j < round(y); j++) {
					cout << "#";
				}
				cout << endl;
			}
		}
	}
};


struct Solver {
	void solve() {
		Random rng(42);

		Input input;
		input.read();

#define SOLVE(max_n) FloorplanState<max_n> floorplan_state; \
		floorplan_state.set_initial_state(input, rng); \
		floorplan_state.calc_score(); \
		SimulatedAnnealing<FloorplanState<max_n>> sa(floorplan_state, rng); \
		sa.optimize<annealing_temperature_schedule>(4.95); \
		Solution solution = floorplan_state.construct_exact_solution<false>(); \
		solution.print();

		if (input.n <= 60) { SOLVE(60) }
		else if (input.n <= 70) { SOLVE(70) }
		else if (input.n <= 80) { SOLVE(80) }
		else if (input.n <= 90) { SOLVE(90) }
		else if (input.n <= 100) { SOLVE(100) }
		else if (input.n <= 110) { SOLVE(110) }
		else if (input.n <= 120) { SOLVE(120) }
		else if (input.n <= 130) { SOLVE(130) }
		else if (input.n <= 140) { SOLVE(140) }
		else if (input.n <= 150) { SOLVE(150) }
		else if (input.n <= 160) { SOLVE(160) }
		else if (input.n <= 170) { SOLVE(170) }
		else if (input.n <= 180) { SOLVE(180) }
		else if (input.n <= 190) { SOLVE(190) }
		else if (input.n <= 200) { SOLVE(200) }

#undef SOLVE

	}
};

int main() {
	/**
	Tester tester;
	//tester.test_initialization();
	//tester.test_gradient_descent();
	//tester.test_initial_state_gradient_descent();
	//tester.test_some_steps();
	//tester.test_monotonic_function();
	/**/

	/**/
	Solver solver;
	solver.solve();
	/**/
	int a = 0;
	//cin >> a;
}

