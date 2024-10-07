C
C @(#)dsort2.f	3.2 (BNP) 12/9/88
C
C QUICK SORT IN FORTRAN
      SUBROUTINE ISORT(N,ARRAY1)
      INTEGER N
      INTEGER ARRAY1(0:N-1)
C
C.... Sort array1 and array2 into increasing order for array1
C
      INTEGER IGAP,I,J
      INTEGER ITEMP
      DOUBLE PRECISION TEMP
C
      IGAP = N/2
 10   IF (IGAP.GT.0) THEN
        DO 200 I = IGAP,N-1
          J = I-IGAP
 50       IF (ARRAY1(J).GT.ARRAY1(J+IGAP)) THEN
              ITEMP = ARRAY1(J)
              ARRAY1(J) = ARRAY1(J+IGAP)
              ARRAY1(J+IGAP) = ITEMP
          ELSE
            GO TO 200
          ENDIF
          IF (J.GE.IGAP) THEN
             J = J-IGAP
             GO TO 50
          ENDIF
 200    CONTINUE
      ELSE
        RETURN
      ENDIF
      IGAP = IGAP/2
      GO TO 10
      END

      SUBROUTINE ISORT2(N,ARRAY1,ARRAY2)
      INTEGER N
      INTEGER ARRAY1(0:N-1)
      INTEGER ARRAY2(0:N-1)
C
C.... Sort array1 and array2 into increasing order for array1
C
      INTEGER IGAP,I,J
      INTEGER ITEMP,JTEMP
      DOUBLE PRECISION TEMP
C
      IGAP = N/2
 10   IF (IGAP.GT.0) THEN
        DO 200 I = IGAP,N-1
          J = I-IGAP
 50       IF (ARRAY1(J).GT.ARRAY1(J+IGAP)) THEN
              ITEMP = ARRAY1(J)
              JTEMP = ARRAY2(J)
              ARRAY1(J) = ARRAY1(J+IGAP)
              ARRAY1(J+IGAP) = ITEMP
              ARRAY2(J) = ARRAY2(J+IGAP)
              ARRAY2(J+IGAP) = JTEMP
          ELSE
            GO TO 200
          ENDIF
          IF (J.GE.IGAP) THEN
             J = J-IGAP
             GO TO 50
          ENDIF
 200    CONTINUE
      ELSE
        RETURN
      ENDIF
      IGAP = IGAP/2
      GO TO 10
      END
C
C QUICK SORT IN FORTRAN
      SUBROUTINE DSORT2(N,ARRAY1,ARRAY2)
      INTEGER N
      DOUBLE PRECISION ARRAY1(0:N-1)
      INTEGER ARRAY2(0:N-1)
C
C.... Sort array1 and array2 into increasing order for array1
C
      INTEGER IGAP,I,J
      INTEGER ITEMP
      DOUBLE PRECISION TEMP
C
      IGAP = N/2
 10   IF (IGAP.GT.0) THEN
        DO 200 I = IGAP,N-1
          J = I-IGAP
 50       IF (ARRAY1(J).GT.ARRAY1(J+IGAP)) THEN
              TEMP = ARRAY1(J)
              ARRAY1(J) = ARRAY1(J+IGAP)
              ARRAY1(J+IGAP) = TEMP
              ITEMP = ARRAY2(J)
              ARRAY2(J) = ARRAY2(J+IGAP)
              ARRAY2(J+IGAP) = ITEMP
          ELSE
            GO TO 200
          ENDIF
          IF (J.GE.IGAP) THEN
             J = J-IGAP
             GO TO 50
          ENDIF
 200    CONTINUE
      ELSE
        RETURN
      ENDIF
      IGAP = IGAP/2
      GO TO 10
      END
C
      SUBROUTINE DSORT2A(N,ARRAY1,ARRAY2)
      INTEGER N
      DOUBLE PRECISION ARRAY1(0:N-1),ARRAY2(0:N-1)
C
C.... Sort array1 and array2 into increasing order for abs(array1)
C
      INTEGER IGAP,I,J
      DOUBLE PRECISION TEMP
C
      IGAP = N/2
 10   IF (IGAP.GT.0) THEN
        DO 200 I = IGAP,N-1
          J = I-IGAP
 50       IF (abs(ARRAY1(J)).GT.abs(ARRAY1(J+IGAP))) THEN
              TEMP = ARRAY1(J)
              ARRAY1(J) = ARRAY1(J+IGAP)
              ARRAY1(J+IGAP) = TEMP
              TEMP = ARRAY2(J)
              ARRAY2(J) = ARRAY2(J+IGAP)
              ARRAY2(J+IGAP) = TEMP
          ELSE
            GO TO 200
          ENDIF
          IF (J.GE.IGAP) THEN
             J = J-IGAP
             GO TO 50
          ENDIF
 200    CONTINUE
      ELSE
        RETURN
      ENDIF
      IGAP = IGAP/2
      GO TO 10
      END

      SUBROUTINE ISORT3(N,ARRAY1,ARRAY2,ARRAY3)
      INTEGER(8) N
      INTEGER(8) ARRAY1(0:N-1),ARRAY2(0:N-1)
      INTEGER ARRAY3(0:N-1)
C
C.... Sort array1 and array2 into increasing order for array1
C
      INTEGER(8) IGAP,I,J
      INTEGER(8) TEMP8
      INTEGER TEMP
C
      IGAP = N/2
 10   IF (IGAP.GT.0) THEN
        DO 200 I = IGAP,N-1
          J = I-IGAP
 50       IF (ARRAY1(J).GT.ARRAY1(J+IGAP)) THEN
              TEMP8 = ARRAY1(J)
              ARRAY1(J) = ARRAY1(J+IGAP)
              ARRAY1(J+IGAP) = TEMP8
              TEMP8 = ARRAY2(J)
              ARRAY2(J) = ARRAY2(J+IGAP)
              ARRAY2(J+IGAP) = TEMP8
              TEMP = ARRAY3(J)
              ARRAY3(J) = ARRAY3(J+IGAP)
              ARRAY3(J+IGAP) = TEMP
          ELSE
            GO TO 200
          ENDIF
          IF (J.GE.IGAP) THEN
             J = J-IGAP
             GO TO 50
          ENDIF
 200    CONTINUE
      ELSE
        RETURN
      ENDIF
      IGAP = IGAP/2
      GO TO 10
      END
C
