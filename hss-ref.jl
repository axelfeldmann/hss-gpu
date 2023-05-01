function matmatup(hssA, B)
    if isleaf(hssA)
        return BinaryNode(hssA.V' * B)
    else
        n1 = hssA.sz1[2]
        Z1 = matmatup(hssA.A11, B[1:n1,:])
        Z2 = matmatup(hssA.A22, B[n1+1:end,:])
        return BinaryNode(hssA.W1' * Z1.data .+ hssA.W2' * Z2.data, Z1, Z2)
    end
end