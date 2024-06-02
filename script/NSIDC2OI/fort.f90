subroutine sst_acspo2oi_1rec(sst_acspo, sst_oi, num_sst_oi, landmask_oi  &
    , idx_lat_oi2acspo, idx_lon_oi2acspo)
    implicit none

    real   (kind=4), dimension(18000, 9000), intent(in ) :: sst_acspo
    real   (kind=4), dimension(1440, 720), intent(out) :: sst_oi
    integer(kind=4), dimension(1440, 720), intent(out) :: num_sst_oi
    integer(kind=4), dimension(1440, 720), intent(in) :: landmask_oi
    integer(kind=4), dimension(2, 720 ), intent(in) :: idx_lat_oi2acspo
    integer(kind=4), dimension(2, 1440), intent(in) :: idx_lon_oi2acspo

! f2py instructions, for f90 not to use Cf2py.
!f2py intent(in) sst_acspo
!f2py intent(out) sst_oi
!f2py intent(out) num_sst_oi
!f2py intent(in) landmask_oi
!f2py intent(in) idx_lat_oi2acspo
!f2py intent(in) idx_lon_oi2acspo

    ! local vars
    integer :: i, j, lon_min, lon_max, lat_min, lat_max

    sst_oi     = 0.0
    num_sst_oi = 0

    do j = 1, 720; do i = 1, 1440
        if (landmask_oi(i, j) == 1) then
            lon_min = idx_lon_oi2acspo(1, i); lon_max = idx_lon_oi2acspo(2, i)
            lat_min = idx_lat_oi2acspo(1, j); lat_max = idx_lat_oi2acspo(2, j)
            sst_oi(i, j)     = sum  ( sst_acspo(lon_min:lon_max, lat_min:lat_max) )
            num_sst_oi(i, j) = count( sst_acspo(lon_min:lon_max, lat_min:lat_max) > 1.0)
        end if
    end do; end do

    return
end subroutine sst_acspo2oi_1rec


subroutine sic_nsidc2oi_nh_1rec(sic_ns, sic_oi, num_sic_oi, landmask_oi, idx_lat_nh_ns2oi, idx_lon_nh_ns2oi)
    implicit none

    real   (kind=4), dimension(304 , 448), intent(in   ) :: sic_ns
    real   (kind=4), dimension(1440, 720), intent(inout) :: sic_oi
    integer(kind=4), dimension(1440, 720), intent(inout) :: num_sic_oi
    integer(kind=4), dimension(1440, 720), intent(in   ) :: landmask_oi
    integer(kind=4), dimension(304 , 448), intent(in   ) :: idx_lat_nh_ns2oi
    integer(kind=4), dimension(304 , 448), intent(in   ) :: idx_lon_nh_ns2oi

! f2py instructions, for f90 not to use Cf2py.
!f2py intent(in   ) sic_ns
!f2py intent(inout) sic_oi
!f2py intent(inout) num_sic_oi
!f2py intent(in   ) landmask_oi
!f2py intent(in   ) idx_lat_nh_ns2oi
!f2py intent(in   ) idx_lon_nh_ns2oi

    ! local vars
    integer :: i, j, idx_lat_oi, idx_lon_oi

  ! sic_oi     = 0.0  ! since we have nh and sh, need to initialize them outside
  ! num_sic_oi = 0

    do j = 1, 448; do i = 1, 304
        if (sic_ns(i, j) <= 0 .or. sic_ns(i, j) > 1) then
            cycle
        end if

        idx_lat_oi = idx_lat_nh_ns2oi(i, j)
        idx_lon_oi = idx_lon_nh_ns2oi(i, j)
 
        if (landmask_oi(idx_lon_oi, idx_lat_oi) == 1) then
            sic_oi(idx_lon_oi, idx_lat_oi) = sic_oi(idx_lon_oi, idx_lat_oi) + sic_ns(i, j)
            num_sic_oi(idx_lon_oi, idx_lat_oi) = num_sic_oi(idx_lon_oi, idx_lat_oi) + 1
        end if
    end do; end do

    return
end subroutine sic_nsidc2oi_nh_1rec


subroutine sic_nsidc2oi_sh_1rec(sic_ns, sic_oi, num_sic_oi, landmask_oi, idx_lat_sh_ns2oi, idx_lon_sh_ns2oi)
    implicit none

    real   (kind=4), dimension(316 , 332), intent(in   ) :: sic_ns
    real   (kind=4), dimension(1440, 720), intent(inout) :: sic_oi
    integer(kind=4), dimension(1440, 720), intent(inout) :: num_sic_oi
    integer(kind=4), dimension(1440, 720), intent(in   ) :: landmask_oi
    integer(kind=4), dimension(316 , 332), intent(in   ) :: idx_lat_sh_ns2oi
    integer(kind=4), dimension(316 , 332), intent(in   ) :: idx_lon_sh_ns2oi

! f2py instructions, for f90 not to use Cf2py.
!f2py intent(in   ) sic_ns
!f2py intent(inout) sic_oi
!f2py intent(inout) num_sic_oi
!f2py intent(in   ) landmask_oi
!f2py intent(in   ) idx_lat_sh_ns2oi
!f2py intent(in   ) idx_lon_sh_ns2oi

    ! local vars
    integer :: i, j, idx_lat_oi, idx_lon_oi

  ! sic_oi     = 0.0  ! since we have nh and sh, need to initialize them outside
  ! num_sic_oi = 0

    do j = 1, 332; do i = 1, 316
        if (sic_ns(i, j) <= 0 .or. sic_ns(i, j) > 1) then
            cycle
        end if

        idx_lat_oi = idx_lat_sh_ns2oi(i, j)
        idx_lon_oi = idx_lon_sh_ns2oi(i, j)
 
        if (landmask_oi(idx_lon_oi, idx_lat_oi) == 1) then
            sic_oi(idx_lon_oi, idx_lat_oi) = sic_oi(idx_lon_oi, idx_lat_oi) + sic_ns(i, j)
            num_sic_oi(idx_lon_oi, idx_lat_oi) = num_sic_oi(idx_lon_oi, idx_lat_oi) + 1
        end if
    end do; end do

    return
end subroutine sic_nsidc2oi_sh_1rec
