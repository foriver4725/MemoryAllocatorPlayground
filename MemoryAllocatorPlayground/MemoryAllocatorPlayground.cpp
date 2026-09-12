#include <cassert>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

class LinearAllocator
{
public:
	explicit LinearAllocator(std::size_t capacity) :
		_memory(std::make_unique<std::byte[]>(capacity)),
		_capacity(capacity)
	{
	}

	template<typename T, typename... Args>
	T* create(Args&&... args)
	{
		void* memory = allocate(sizeof(T), alignof(T));
		return new (memory) T{ std::forward<Args>(args)... };
	}

	template<typename T>
	void destroy(T* object) noexcept
	{
		if (object)
		{
			std::destroy_at(object);
		}
	}

	// 線形アロケータでは個別にメモリを解放しない。
	void reset() noexcept
	{
		_offset = 0;
	}

	std::size_t used() const noexcept { return _offset; }
	std::size_t capacity() const noexcept { return _capacity; }

private:
	std::unique_ptr<std::byte[]> _memory;
	std::size_t _capacity{};
	std::size_t _offset{};

	static constexpr std::size_t get_aligned(std::size_t value, std::size_t alignment) noexcept
	{
		return (value + alignment - 1) & ~(alignment - 1);
	}

	void* allocate(std::size_t size, std::size_t alignment)
	{
		if (alignment == 0 || (alignment & (alignment - 1)) != 0)
		{
			throw std::invalid_argument{ "alignment must be a power of two" };
		}

		const std::size_t current_address = reinterpret_cast<std::size_t>(_memory.get() + _offset);
		const std::size_t aligned_address = get_aligned(current_address, alignment);
		const std::size_t padding = aligned_address - current_address;
		const std::size_t remaining = _capacity - _offset;

		if (remaining < size ||
			remaining - size < padding)
		{
			throw std::bad_alloc{};
		}

		void* result = _memory.get() + _offset + padding;
		_offset += padding + size;

		return result;
	}
};

#pragma region Test

struct Particle {
	float x;
	float y;

	Particle(float x_value, float y_value)
		: x(x_value),
		y(y_value)
	{
	}
};

// 64バイト境界を必要とするテスト用の型。
struct alignas(64) AlignedObject {
	int value;
};

// デストラクタが呼ばれた回数を記録する型。
struct LifetimeObserver {
	int* destruction_count;

	explicit LifetimeObserver(int* count)
		: destruction_count(count)
	{
	}

	~LifetimeObserver()
	{
		++(*destruction_count);
	}
};

void test_initial_state()
{
	LinearAllocator arena{ 1024 };

	assert(arena.capacity() == 1024);
	assert(arena.used() == 0);
}

void test_create()
{
	LinearAllocator arena{ 1024 };

	auto* particle =
		arena.create<Particle>(1.5f, 2.5f);

	assert(particle != nullptr);
	assert(particle->x == 1.5f);
	assert(particle->y == 2.5f);
	assert(arena.used() >= sizeof(Particle));

	arena.destroy(particle);
}

void test_destroy()
{
	LinearAllocator arena{ 1024 };
	int destruction_count = 0;

	auto* object =
		arena.create<LifetimeObserver>(
			&destruction_count
		);

	assert(destruction_count == 0);

	const std::size_t used_before_destroy =
		arena.used();

	arena.destroy(object);

	assert(destruction_count == 1);

	// destroyはデストラクタを呼ぶだけなので、
	// 使用済みメモリ量は変化しない。
	assert(arena.used() == used_before_destroy);

	arena.reset();

	assert(arena.used() == 0);
}

void test_multiple_create()
{
	LinearAllocator arena{ 1024 };

	auto* first =
		arena.create<Particle>(1.0f, 2.0f);

	const std::size_t used_after_first =
		arena.used();

	auto* second =
		arena.create<Particle>(3.0f, 4.0f);

	assert(first != second);
	assert(first->x == 1.0f);
	assert(first->y == 2.0f);
	assert(second->x == 3.0f);
	assert(second->y == 4.0f);

	assert(arena.used() > used_after_first);

	arena.destroy(second);
	arena.destroy(first);
}

void test_alignment()
{
	LinearAllocator arena{ 1024 };

	// 先に1バイト確保し、現在位置を意図的にずらす。
	arena.create<std::byte>();

	auto* object =
		arena.create<AlignedObject>();

	const auto address =
		reinterpret_cast<std::uintptr_t>(object);

	assert(address % alignof(AlignedObject) == 0);
	assert(object->value == 0);

	arena.destroy(object);
}

void test_reset()
{
	LinearAllocator arena{ 1024 };

	auto* particle =
		arena.create<Particle>(1.0f, 2.0f);

	assert(arena.used() > 0);

	// resetより前にデストラクタを呼ぶ。
	arena.destroy(particle);

	arena.reset();

	assert(arena.used() == 0);
}

void test_reuse_after_reset()
{
	LinearAllocator arena{ 1024 };

	auto* first =
		arena.create<Particle>(1.0f, 2.0f);

	const void* first_address = first;

	arena.destroy(first);
	arena.reset();

	auto* second =
		arena.create<Particle>(3.0f, 4.0f);

	const void* second_address = second;

	// 同じ確保順なら、リセット後は同じ場所を再利用する。
	assert(first_address == second_address);
	assert(second->x == 3.0f);
	assert(second->y == 4.0f);

	arena.destroy(second);
}

void test_out_of_memory()
{
	LinearAllocator arena{
		sizeof(Particle) + alignof(Particle) - 1
	};

	auto* first =
		arena.create<Particle>(1.0f, 2.0f);

	bool bad_alloc_caught = false;

	try
	{
		arena.create<Particle>(3.0f, 4.0f);
	}
	catch (const std::bad_alloc&)
	{
		bad_alloc_caught = true;
	}

	assert(bad_alloc_caught);

	arena.destroy(first);
}

void test_destructor()
{
	LinearAllocator arena{ 1024 };
	int destruction_count = 0;

	auto* object =
		arena.create<LifetimeObserver>(
			&destruction_count
		);

	assert(destruction_count == 0);

	arena.destroy(object);

	assert(destruction_count == 1);

	arena.reset();

	assert(arena.used() == 0);
	assert(destruction_count == 1);
}

int main()
{
	test_initial_state();
	test_create();
	test_destroy();
	test_multiple_create();
	test_alignment();
	test_reset();
	test_reuse_after_reset();
	test_out_of_memory();
	test_destructor();

	std::cout << "All tests passed!\n";

	return 0;
}
#pragma endregion
