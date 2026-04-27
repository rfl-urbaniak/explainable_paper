from fractions import Fraction

probs = {}

probs[1, 1, 1] = Fraction("0.5") * Fraction("0.5") * Fraction("0.9")
probs[1, 1, 0] = Fraction("0.5") * Fraction("0.5") - probs[1, 1, 1]

probs[1, 0, 1] = Fraction("0.5") * Fraction("0.5") * Fraction("0.9") * Fraction("0.05")
probs[1, 0, 0] = Fraction("0.5") * Fraction("0.5") - probs[1, 0, 1]

probs[0, 1, 1] = Fraction("0.5") * Fraction("0.5") * Fraction("0.2") * Fraction("0.9")
probs[0, 1, 0] = Fraction("0.5") * Fraction("0.5") - probs[0, 1, 1]

probs[0, 0, 1] = Fraction("0")
probs[0, 0, 0] = Fraction("0.5") * Fraction("0.5")

assert sum(probs.values()) == Fraction("1")


def prob(gender=None, credit=None, loan=None):
    gender = [gender] if gender is not None else [0, 1]
    credit = [credit] if credit is not None else [0, 1]
    loan = [loan] if loan is not None else [0, 1]

    return sum(probs[g, c, l] for g in gender for c in credit for l in loan)


p_credit_1__gender_0_loan_0 = prob(credit=1, gender=0, loan=0) / prob(gender=0, loan=0)
p_credit_1__gender_1_loan_0 = prob(credit=1, gender=1, loan=0) / prob(gender=1, loan=0)
p_credit_1__gender_0_loan_1 = prob(credit=1, gender=0, loan=1) / prob(gender=0, loan=1)
p_credit_1__gender_1_loan_1 = prob(credit=1, gender=1, loan=1) / prob(gender=1, loan=1)

p_gender_1__credit_0_loan_0 = prob(credit=0, gender=1, loan=0) / prob(credit=0, loan=0)
p_gender_1__credit_1_loan_1 = prob(credit=1, gender=1, loan=1) / prob(credit=1, loan=1)

pn_gender_1_loan_0 = (
    p_credit_1__gender_1_loan_0 * Fraction("0.2") * (1 - Fraction("0.1"))
)


pn_credit_0_loan_0 = p_gender_1__credit_0_loan_0 * Fraction("0.9") + (
    1 - p_gender_1__credit_0_loan_0
) * Fraction("0.2") * Fraction("0.1")


pn_gender_0_loan_0 = p_credit_1__gender_0_loan_0 * Fraction("0.9") + (
    1 - p_credit_1__gender_0_loan_0
) * Fraction("0.9") * Fraction("0.05")

print("PN(gender=1, loan=0)", float(pn_gender_1_loan_0), pn_gender_1_loan_0)
print("PN(gender=0, loan=0)", float(pn_gender_0_loan_0), pn_gender_0_loan_0)
print("PN(credit=0, loan=0)", float(pn_credit_0_loan_0), pn_credit_0_loan_0)


ps_gender_1_loan_0 = p_credit_1__gender_0_loan_1 * Fraction("0.1") + (
    1 - p_credit_1__gender_0_loan_1
) * (Fraction("0.1") + Fraction("0.9") * (1 - Fraction("0.05")))

ps_gender_0_loan_0 = p_credit_1__gender_1_loan_1 * (
    Fraction("0.8") + Fraction("0.2") * Fraction("0.1")
) + (1 - p_credit_1__gender_1_loan_1)

ps_credit_0_loan_0 = p_gender_1__credit_1_loan_1 * (
    Fraction("0.1") + Fraction("0.9") * (1 - Fraction("0.05"))
) + (1 - p_gender_1__credit_1_loan_1)

print("PS(gender=1, loan=0)", float(ps_gender_1_loan_0))
print("PS(gender=0, loan=0)", float(ps_gender_0_loan_0))
print("PS(credit=0, loan=0)", float(ps_credit_0_loan_0))


pns_gender_1_loan_0 = (
    Fraction("0.5")  # credit = 1
    * Fraction("0.1")  # do(M, gender=1) = 0
    * (Fraction("0.2") * (1 - Fraction("0.1")))  # do(M, gender=0) = 1
) + (
    Fraction("0.5")  # credit = 0
    * (
        Fraction("0.1") + Fraction("0.9") * (1 - Fraction("0.05"))
    )  # do(M, gender=1) = 0
    * Fraction("0")  # do(M, gender=0) = 1
)

pns_gender_0_loan_0 = (
    Fraction("0.5")  # credit = 1
    * (Fraction("0.8") + Fraction("0.2") * Fraction("0.1"))  # do(M, gender=0) = 0
    * Fraction("0.9")  # do(M, gender=1) = 1
) + (
    Fraction("0.5")  # credit = 0
    * 1  # do(M, gender=0) = 0
    * (Fraction("0.9") * Fraction("0.05"))  # do(M, gender=1) = 1
)

pns_credit_0_loan_0 = (
    Fraction("0.5")  # gender=1
    * (
        Fraction("0.1") + Fraction("0.9") * (1 - Fraction("0.05"))
    )  # do(M, credit=0) = 0
    * (Fraction("0.9"))  # do(M, credit=1) = 1
) + (
    Fraction("0.5")  # gender = 0
    * Fraction("1")  # do(M, credit=0) = 0
    * (Fraction("0.2") * (1 - Fraction("0.1")))  # do(M, credit=1) = 1
)

print("PNS(gender=1, loan=0)", float(pns_gender_1_loan_0))
print("PNS(gender=0, loan=0)", float(pns_gender_0_loan_0))
print("PNS(credit=0, loan=0)", float(pns_credit_0_loan_0))
