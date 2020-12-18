# include("unit_test.jl")

using Test

function test_tree(T::tf.Tree)
    @test T.leaves ⊆ T.nodes
    @test T.branches ⊆ T.nodes
    @test vcat(T.leaves,T.branches) ⊆ T.nodes
    @test T.nodes ⊆ vcat(T.leaves,T.branches)
    @test unique(values(T.z)) ⊆ T.leaves
    @test T.leaves ⊆ unique(values(T.z))
    @test unique(values(T.z)) ⊈ T.branches
    @test keys(T.b) ⊆ T.branches
    @test keys(T.a) ⊆ T.branches
    @test length(T.nodes) == length(unique(T.nodes))
    @test length(T.leaves) == length(unique(T.leaves))
    @test length(T.branches) == length(unique(T.branches))
    if isempty(T.a) == false
        @test keys(T.a) ⊈ T.leaves
        @test keys(T.b) ⊈ T.leaves
    end
    @test isempty(T.z) == false
end
